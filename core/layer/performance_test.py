#!/usr/bin/env python3
"""
性能测试脚本 - 比较GPTQ融合算子和原始实现
"""
import torch
import time
import numpy as np
import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layer.qwen_layer import QwenModelLayerAdapter
from core.layer.optimized_qwen_layer import OptimizedQwenModelLayerAdapter
from core.layer.gptq import GPTQTritonFusion

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_quantized_layer():
    """创建模拟的量化层"""
    class MockQuantizedLayer:
        def __init__(self):
            # 模拟GPTQ量化参数
            self.attn = MockQuantizedAttention()
            self.ln_1 = torch.nn.LayerNorm(4096)
            self.ln_2 = torch.nn.LayerNorm(4096)
            self.mlp = torch.nn.Linear(4096, 4096 * 4)
    
    class MockQuantizedAttention:
        def __init__(self):
            self.c_attn = MockQuantizedLinear(4096, 4096 * 3)
            self.c_proj = MockQuantizedLinear(4096, 4096)
    
    class MockQuantizedLinear:
        def __init__(self, in_features, out_features):
            # 模拟GPTQ量化权重
            self.qweight = torch.randint(0, 256, (out_features, in_features // 8), 
                                       dtype=torch.int32, device='cuda')
            self.scales = torch.randn(out_features // 128, in_features, 
                                    dtype=torch.float16, device='cuda')
            self.qzeros = torch.randint(0, 16, (out_features // 128, in_features // 8), 
                                      dtype=torch.int32, device='cuda')
    
    return MockQuantizedLayer()


def benchmark_layer_processing():
    """基准测试层处理性能"""
    print("🚀 开始性能基准测试...")
    
    # 测试参数
    batch_size = 4
    seq_len = 512
    hidden_dim = 4096
    num_heads = 32
    head_size = 128
    num_layers = 10
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, 
                               device='cuda', dtype=torch.float16)
    
    # 创建模拟层
    mock_layer = create_mock_quantized_layer()
    
    # 创建适配器
    model_config = type('Config', (), {'group_size': 128})()
    
    # 原始适配器
    original_adapter = QwenModelLayerAdapter(
        model_config=model_config,
        device='cuda',
        num_heads=num_heads,
        head_size=head_size
    )
    
    # 优化适配器
    optimized_adapter = OptimizedQwenModelLayerAdapter(
        model_config=model_config,
        device='cuda',
        num_heads=num_heads,
        head_size=head_size
    )
    
    # 模拟缓存管理器
    class MockCacheManager:
        def get(self, layer_idx):
            return None, None
    
    cache_manager = MockCacheManager()
    seq_ids = list(range(batch_size))
    context_lens = [seq_len] * batch_size
    
    # 预热
    print("🔥 预热阶段...")
    for _ in range(5):
        with torch.no_grad():
            _ = original_adapter.process_layer(
                mock_layer, hidden_states, cache_manager, 
                seq_ids, context_lens, layer_idx=0
            )
            _ = optimized_adapter.process_layer(
                mock_layer, hidden_states, cache_manager, 
                seq_ids, context_lens, layer_idx=0
            )
    
    torch.cuda.synchronize()
    
    # 测试原始实现
    print("📊 测试原始实现...")
    original_times = []
    for i in range(20):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            result = original_adapter.process_layer(
                mock_layer, hidden_states, cache_manager, 
                seq_ids, context_lens, layer_idx=i % num_layers
            )
        
        torch.cuda.synchronize()
        original_times.append(time.time() - start_time)
    
    # 测试优化实现
    print("📊 测试优化实现...")
    optimized_times = []
    for i in range(20):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            result = optimized_adapter.process_layer(
                mock_layer, hidden_states, cache_manager, 
                seq_ids, context_lens, layer_idx=i % num_layers
            )
        
        torch.cuda.synchronize()
        optimized_times.append(time.time() - start_time)
    
    # 计算统计信息
    original_avg = np.mean(original_times) * 1000
    original_std = np.std(original_times) * 1000
    optimized_avg = np.mean(optimized_times) * 1000
    optimized_std = np.std(optimized_times) * 1000
    
    speedup = original_avg / optimized_avg
    
    # 打印结果
    print("\n" + "="*60)
    print("📈 性能测试结果")
    print("="*60)
    print(f"原始实现:     {original_avg:.2f} ± {original_std:.2f} ms")
    print(f"优化实现:     {optimized_avg:.2f} ± {optimized_std:.2f} ms")
    print(f"加速比:       {speedup:.2f}x")
    print(f"性能提升:     {(speedup - 1) * 100:.1f}%")
    print("="*60)
    
    return {
        'original_avg': original_avg,
        'optimized_avg': optimized_avg,
        'speedup': speedup
    }


def benchmark_gptq_fusion():
    """测试GPTQ融合算子性能"""
    print("\n🔥 测试GPTQ融合算子性能...")
    
    # 测试参数
    M, N, K = 512, 2048, 2048
    groupsize = 128
    num_warmup = 5
    num_iterations = 20
    
    # 创建测试数据
    input = torch.randn((M, K), dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (K, N // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn((num_groups, N), dtype=torch.float16, device='cuda')
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 预热
    print("🔥 预热GPTQ融合算子...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = gptq_fusion.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            _ = gptq_fusion.baseline_gptq_gemm(input, qweight, qzeros, scales, groupsize)
    
    torch.cuda.synchronize()
    
    # 测试GPTQ融合算子
    print("📊 测试GPTQ融合算子...")
    gptq_times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            result = gptq_fusion.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
        
        torch.cuda.synchronize()
        gptq_times.append(time.time() - start)
    
    # 测试基线实现
    print("📊 测试基线实现...")
    baseline_times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            result = gptq_fusion.baseline_gptq_gemm(input, qweight, qzeros, scales, groupsize)
        
        torch.cuda.synchronize()
        baseline_times.append(time.time() - start)
    
    # 计算统计信息
    gptq_avg = np.mean(gptq_times) * 1000
    gptq_std = np.std(gptq_times) * 1000
    baseline_avg = np.mean(baseline_times) * 1000
    baseline_std = np.std(baseline_times) * 1000
    
    speedup = baseline_avg / gptq_avg
    
    print(f"\n📈 GPTQ融合算子性能结果:")
    print(f"   矩阵大小: {M}x{N}x{K}")
    print(f"   GPTQ融合: {gptq_avg:.2f}±{gptq_std:.2f} ms")
    print(f"   基线实现: {baseline_avg:.2f}±{baseline_std:.2f} ms")
    print(f"   加速比: {speedup:.2f}x")
    print(f"   性能提升: {(speedup - 1) * 100:.1f}%")
    
    return {
        'gptq_avg': gptq_avg,
        'baseline_avg': baseline_avg,
        'speedup': speedup
    }


def analyze_bottlenecks():
    """分析性能瓶颈"""
    print("\n🔍 性能瓶颈分析...")
    
    bottlenecks = [
        "1. 量化检测开销 - 每层重复检测量化状态",
        "2. 量化参数获取开销 - 重复获取qweight/scales/qzeros",
        "3. Triton内核启动开销 - 每次调用重新编译内核",
        "4. 内存分配开销 - 每次调用重新分配输出张量",
        "5. 缺少内核预热 - Triton内核需要预热才能达到最佳性能"
    ]
    
    optimizations = [
        "✅ 缓存量化状态 - 避免重复检测",
        "✅ 缓存量化参数 - 避免重复获取",
        "✅ 使用GPTQ融合算子 - 减少内存访问",
        "✅ 预分配输出张量 - 减少内存分配",
        "✅ 添加内核预热 - 提升Triton性能"
    ]
    
    print("🚨 主要性能瓶颈:")
    for bottleneck in bottlenecks:
        print(f"   {bottleneck}")
    
    print("\n🚀 优化措施:")
    for optimization in optimizations:
        print(f"   {optimization}")


def main():
    """主函数"""
    print("🎯 LLM推理框架性能优化测试")
    print("="*60)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return
    
    try:
        # 分析瓶颈
        analyze_bottlenecks()
        
        # 测试GPTQ融合算子
        gptq_results = benchmark_gptq_fusion()
        
        # 运行层处理基准测试
        layer_results = benchmark_layer_processing()
        
        # 总结
        print(f"\n🎉 测试完成！")
        print("="*60)
        print("📊 性能测试总结:")
        print(f"   GPTQ融合算子: {gptq_results['speedup']:.2f}x加速")
        print(f"   层处理优化: {layer_results['speedup']:.2f}x加速")
        
        if gptq_results['speedup'] > 1.5:
            print(f"✅ GPTQ融合算子效果显著：{gptq_results['speedup']:.2f}x加速")
        elif gptq_results['speedup'] > 1.1:
            print(f"✅ GPTQ融合算子有效果：{gptq_results['speedup']:.2f}x加速")
        else:
            print(f"⚠️  GPTQ融合算子效果有限：{gptq_results['speedup']:.2f}x加速")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
