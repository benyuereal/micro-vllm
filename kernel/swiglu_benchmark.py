#!/usr/bin/env python3
# kernel/swiglu_benchmark_qwen7b.py
"""
SwiGLU Qwen-7B 专项性能基准测试
- 维度: H=4096, I=11008 (Qwen-7B 标准)
- 覆盖 Batch Size: 1~128 (模拟 2048 seq_len 场景)
- 对比: PyTorch 原生 vs 分离架构优化版
"""
import torch
import torch.nn.functional as F
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swiglu import fused_swiglu


class Qwen7BSwiGLUBenchmark:
    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        # Qwen-7B 标准维度
        self.H = 4096      # hidden_size
        self.I = 11008     # intermediate_size (4096 * 2.6875)
        self.seq_len = 2048  # 标准序列长度
        
        print(f"🚀 Qwen-7B SwiGLU Benchmark")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Dtype: {dtype}")
        print(f"   Config: H={self.H}, I={self.I}, SeqLen={self.seq_len}")
        print("=" * 80)
    
    def _generate_weights(self):
        """生成符合真实分布的 Qwen-7B 权重"""
        # Qwen 使用 std=0.02 的初始化
        scale = 0.02
        
        gate_w = torch.randn((self.I, self.H), device=self.device, dtype=self.dtype) * scale
        up_w = torch.randn((self.I, self.H), device=self.device, dtype=self.dtype) * scale
        down_w = torch.randn((self.H, self.I), device=self.device, dtype=self.dtype) * scale
        
        return gate_w, up_w, down_w
    
    def _generate_input(self, batch_size):
        """生成输入: [batch, seq_len, hidden] -> flatten to [M, H]"""
        # 模拟经过 RMSNorm 后的输入 (std≈1.0)
        x = torch.randn(
            (batch_size, self.seq_len, self.H), 
            device=self.device, 
            dtype=torch.float32
        )
        x = x / (x.std() + 1e-6)  # 标准化
        return x.to(self.dtype)
    
    def _pytorch_native(self, x, gate_w, up_w, down_w):
        """PyTorch 原生实现 (3 个独立 kernel)"""
        # x: [M, H]
        gate = F.linear(x, gate_w)      # [M, I]
        up = F.linear(x, up_w)          # [M, I]
        hidden = up * F.silu(gate.float()).half()  # [M, I]
        return F.linear(hidden, down_w)  # [M, H]
    
    def _benchmark_impl(self, func, x, gate_w, up_w, down_w, num_iter=50, warmup=10):
        """精确计时"""
        # 预热
        for _ in range(warmup):
            _ = func(x, gate_w, up_w, down_w)
        torch.cuda.synchronize()
        
        # 计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iter):
            _ = func(x, gate_w, up_w, down_w)
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / num_iter  # ms
    
    def run(self, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128], num_iter=50):
        """运行全量测试"""
        results = []
        total_params = (self.I * self.H * 2 + self.H * self.I) / 1e6  # 约 90M params
        
        print(f"\n📊 开始测试 (Params: ~{total_params:.1f}M)")
        print(f"{'Batch':<8} {'Seqs':<8} {'M':<10} {'PyTorch(ms)':<14} {'Optimized(ms)':<16} {'Speedup':<10} {'Status':<8}")
        print("-" * 80)
        
        gate_w, up_w, down_w = self._generate_weights()
        
        for batch in batch_sizes:
            M = batch * self.seq_len  # 总 token 数
            
            try:
                # 生成输入
                x = self._generate_input(batch)
                x_flat = x.view(-1, self.H)  # [M, H]
                
                # 检查数值正确性 (仅第一次)
                if batch == batch_sizes[0]:
                    y_opt = fused_swiglu(x_flat, gate_w, up_w, down_w)
                    y_ref = self._pytorch_native(x_flat, gate_w, up_w, down_w)
                    max_err = torch.max(torch.abs(y_opt - y_ref)).item()
                    status = "✅" if max_err < 0.01 else "❌"
                else:
                    status = "-"
                
                # 性能测试
                torch_time = self._benchmark_impl(
                    self._pytorch_native, x_flat, gate_w, up_w, down_w, num_iter
                )
                opt_time = self._benchmark_impl(
                    fused_swiglu, x_flat, gate_w, up_w, down_w, num_iter
                )
                
                speedup = torch_time / opt_time
                
                results.append({
                    'batch': batch,
                    'M': M,
                    'torch': torch_time,
                    'opt': opt_time,
                    'speedup': speedup,
                    'error': max_err if batch == batch_sizes[0] else 0
                })
                
                print(f"{batch:<8} {batch*self.seq_len:<8} {M:<10} {torch_time:<14.3f} {opt_time:<16.3f} {speedup:<10.2f}x {status:<8}")
                
                # 显存清理
                del x, x_flat
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"{batch:<8} {batch*self.seq_len:<8} {M:<10} {'OOM':<14} {'OOM':<16} {'-':<10} {'⚠️':<8}")
                break
        
        return results
    
    def analyze(self, results):
        """深度分析"""
        if not results:
            return
        
        print("\n" + "=" * 80)
        print("📈 性能分析报告")
        print("=" * 80)
        
        # 1. 加速比趋势
        print("\n1. 加速比趋势:")
        for r in results:
            bar = "█" * int(r['speedup'] * 10)
            print(f"   Batch={r['batch']:3d}: {r['speedup']:.2f}x {bar}")
        
        # 2. 平均加速
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)
        min_speedup = min(r['speedup'] for r in results)
        
        print(f"\n2. 统计摘要:")
        print(f"   平均加速: {avg_speedup:.2f}x")
        print(f"   最大加速: {max_speedup:.2f}x (Batch={[r for r in results if r['speedup']==max_speedup][0]['batch']})")
        print(f"   最小加速: {min_speedup:.2f}x (Batch={[r for r in results if r['speedup']==min_speedup][0]['batch']})")
        
        # 3. 显存节省分析
        print(f"\n3. 显存占用分析 (Batch={results[-1]['batch']}):")
        max_m = results[-1]['M']
        hidden_size = max_m * self.I * 2 / 1024**3  # GB
        print(f"   Hidden 激活显存: {hidden_size:.2f} GB (M={max_m}, I={self.I})")
        print(f"   说明: 分离架构需额外存储 hidden，但计算更快")
        
        # 4. 端到端收益估算
        print(f"\n4. 端到端收益估算:")
        mlp_ratio = 0.31  # MLP 占 Transformer 层 31%
        end2end_gain = (avg_speedup - 1) * mlp_ratio * 100
        print(f"   MLP 占比: {mlp_ratio*100:.0f}%")
        print(f"   预期端到端加速: +{end2end_gain:.1f}%")
        print(f"   1000 tokens 推理: 节省 ~{end2end_gain*0.1:.1f}ms (假设原延迟 100ms)")
        
        # 5. 最佳配置建议
        best = max(results, key=lambda x: x['speedup'])
        print(f"\n5. 最佳配置:")
        print(f"   Batch Size: {best['batch']} (M={best['M']})")
        print(f"   加速比: {best['speedup']:.2f}x")
        print(f"   延迟: {best['opt']:.2f}ms vs {best['torch']:.2f}ms")
        
        # 6. 数值正确性
        if results[0]['error'] < 0.01:
            print(f"\n✅ 数值正确性: 通过 (Max Error={results[0]['error']:.6f} < 0.01)")
        else:
            print(f"\n❌ 数值正确性: 失败 (Max Error={results[0]['error']:.6f})")
        
        print("=" * 80)


def main():
    # 设置 GPU 性能模式
    torch.backends.cudnn.benchmark = True
    
    benchmark = Qwen7BSwiGLUBenchmark()
    
    # 测试配置 (从单条到满负载)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    results = benchmark.run(batch_sizes, num_iter=50)
    benchmark.analyze(results)
    
    # 保存结果
    try:
        import json
        with open('swiglu_qwen7b_results.json', 'w') as f:
            json.dump([{
                'batch': r['batch'],
                'M': r['M'],
                'pytorch_ms': r['torch'],
                'optimized_ms': r['opt'],
                'speedup': r['speedup']
            } for r in results], f, indent=2)
        print(f"\n💾 结果已保存到 swiglu_qwen7b_results.json")
    except Exception as e:
        pass
    
    print("\n🎉 Qwen-7B SwiGLU 基准测试完成!")


if __name__ == "__main__":
    # 检查 swiglu 是否可用
    try:
        from swiglu import fused_swiglu
    except ImportError as e:
        print(f"❌ 错误: 无法导入 swiglu.py - {e}")
        print("   请确保 swiglu.py 在同目录下")
        sys.exit(1)
    
    main()