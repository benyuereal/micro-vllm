#!/usr/bin/env python3
"""
融合内核性能测试 - 目标0.25ms延迟
包含功能验证、性能测试和对比分析
"""

import torch
from torch.utils.cpp_extension import load
import time
import numpy as np

def test_functionality(kernel_module, input_tensor, qweight, qzeros, scales, ln_weight, ln_bias, 
                      batch_size, seq_len, hidden_dim, groupsize, eps):
    """功能验证测试"""
    print("🔍 功能验证测试...")
    
    try:
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
        
        q_output = qkv_output[0]
        k_output = qkv_output[1] 
        v_output = qkv_output[2]
        
        # 检查输出形状
        expected_shape = (batch_size, seq_len, hidden_dim)
        if q_output.shape != expected_shape or k_output.shape != expected_shape or v_output.shape != expected_shape:
            print(f"❌ 输出形状检查失败，期望: {expected_shape}")
            return False
        
        # 检查输出非零
        if q_output.sum().item() == 0 or k_output.sum().item() == 0 or v_output.sum().item() == 0:
            print("❌ 输出非零检查失败")
            return False
        
        # 检查数值稳定性
        if torch.isnan(q_output).any() or torch.isinf(q_output).any():
            print("❌ Q输出数值检查失败")
            return False
        
        if torch.isnan(k_output).any() or torch.isinf(k_output).any():
            print("❌ K输出数值检查失败")
            return False
        
        if torch.isnan(v_output).any() or torch.isinf(v_output).any():
            print("❌ V输出数值检查失败")
            return False
        
        print("✅ 功能验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 功能验证失败: {e}")
        return False

def test_performance(kernel_module, input_tensor, qweight, qzeros, scales, ln_weight, ln_bias, 
                    batch_size, seq_len, hidden_dim, groupsize, eps, num_runs=200):
    """性能测试"""
    print("⏱️ 性能测试...")
    
    # 预热
    print("🔄 预热中...")
    for _ in range(20):
        kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
    
    # 性能测试
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for i in range(num_runs):
        start_event.record()
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
        
        if i % 50 == 0:
            print(f"🔄 运行 {i+1}/{num_runs}, 当前延迟: {elapsed_time:.3f}ms")
    
    return np.array(times)

def main():
    print("🚀 开始融合内核性能测试...")
    print("🎯 目标延迟: < 0.25ms")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"📊 PyTorch版本: {torch.__version__}")
    
    try:
        # 编译内核
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math", "--ptxas-options=-v", "--maxrregcount=255"],
            verbose=True
        )
        
        print("✅ 编译成功!")
        
        # 性能测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        groupsize = 128
        eps = 1e-5
        
        print(f"📊 测试配置: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        print(f"📊 groupsize={groupsize}, eps={eps}")
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        print(f"📊 输入形状: {input_tensor.shape}")
        print(f"📊 qweight形状: {qweight.shape}")
        print(f"📊 qzeros形状: {qzeros.shape}")
        print(f"📊 scales形状: {scales.shape}")
        
        # 功能验证测试
        if not test_functionality(kernel_module, input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                                 batch_size, seq_len, hidden_dim, groupsize, eps):
            return False
        
        # 性能测试
        times = test_performance(kernel_module, input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                                batch_size, seq_len, hidden_dim, groupsize, eps, num_runs=200)
        
        # 统计结果
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # 计算百分位数
        p50 = np.percentile(times, 50)
        p90 = np.percentile(times, 90)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        print("\n📊 性能测试结果:")
        print(f"  平均延迟: {avg_time:.3f}ms")
        print(f"  最小延迟: {min_time:.3f}ms")
        print(f"  最大延迟: {max_time:.3f}ms")
        print(f"  标准差: {std_time:.3f}ms")
        print(f"  P50延迟: {p50:.3f}ms")
        print(f"  P90延迟: {p90:.3f}ms")
        print(f"  P95延迟: {p95:.3f}ms")
        print(f"  P99延迟: {p99:.3f}ms")
        
        # 性能评估
        target_time = 0.25  # 目标0.25ms
        print(f"\n🎯 目标评估 (目标: {target_time}ms):")
        
        if avg_time <= target_time:
            print(f"✅ 平均延迟达标! {avg_time:.3f}ms <= {target_time}ms")
        else:
            print(f"⚠️ 平均延迟未达标! {avg_time:.3f}ms > {target_time}ms")
        
        if p95 <= target_time:
            print(f"✅ P95延迟达标! {p95:.3f}ms <= {target_time}ms")
        else:
            print(f"⚠️ P95延迟未达标! {p95:.3f}ms > {target_time}ms")
        
        if p99 <= target_time * 1.2:  # 允许20%的波动
            print(f"✅ P99延迟达标! {p99:.3f}ms <= {target_time * 1.2:.3f}ms")
        else:
            print(f"⚠️ P99延迟未达标! {p99:.3f}ms > {target_time * 1.2:.3f}ms")
        
        # 性能提升分析
        if avg_time < 0.3:
            improvement = (0.3 - avg_time) / 0.3 * 100
            print(f"🚀 相比0.3ms目标提升: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
