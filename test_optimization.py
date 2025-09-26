#!/usr/bin/env python3
"""
测试GPTQ优化效果
"""
import torch
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def test_optimization():
    """测试优化效果"""
    print("🚀 测试GPTQ优化效果...")
    
    # 使用中等大小的矩阵
    M, K, N = 256, 1024, 512
    groupsize = 128
    
    # 创建测试数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"测试矩阵大小: {M}x{K}x{N}")
    print(f"qweight形状: {qweight.shape}")
    print(f"qzeros形状: {qzeros.shape}")
    print(f"scales形状: {scales.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 测试不同的反量化方法
    methods = {
        'vectorized': fusion.dequantize_gptq_weight,
        'batch_optimized': fusion.dequantize_gptq_weight_batch_optimized,
        'ultra_optimized': fusion.dequantize_gptq_weight_ultra_optimized
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n🧪 测试 {method_name} 方法...")
        
        # 预热
        for _ in range(3):
            _ = method_func(qweight, qzeros, scales, groupsize)
        torch.cuda.synchronize()
        
        # 测试性能
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            dequantized_weight = method_func(qweight, qzeros, scales, groupsize)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
        results[method_name] = avg_time
        print(f"✅ {method_name}: {avg_time:.2f} ms")
        
        # 测试基线实现
        print(f"🐌 测试 {method_name} 基线实现...")
        torch.cuda.synchronize()
        start = time.time()
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        torch.cuda.synchronize()
        baseline_time = time.time() - start
        print(f"✅ 基线实现: {baseline_time*1000:.2f} ms")
        
        # 比较结果
        result = torch.matmul(input_tensor, dequantized_weight.T)
        diff = torch.abs(result - baseline_result).max()
        print(f"📊 最大差异: {diff.item():.6f}")
        
        if diff < 1e-3:
            print("🎉 结果一致！")
        else:
            print("⚠️  结果有差异")
    
    # 计算加速比
    print(f"\n📈 性能比较:")
    baseline_time = results['vectorized']
    for method_name, time_ms in results.items():
        speedup = baseline_time / time_ms
        print(f"{method_name}: {time_ms:.2f} ms (加速比: {speedup:.2f}x)")

def test_small_matrix():
    """测试小矩阵的优化效果"""
    print("\n🧪 测试小矩阵优化效果...")
    
    # 使用小矩阵
    M, K, N = 32, 256, 128
    groupsize = 128
    
    # 创建测试数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"小矩阵大小: {M}x{K}x{N}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 测试向量化方法
    print("🚀 测试向量化方法...")
    start = time.time()
    dequantized_weight = fusion.dequantize_gptq_weight(qweight, qzeros, scales, groupsize)
    vectorized_time = time.time() - start
    
    # 测试批量优化方法
    print("🚀 测试批量优化方法...")
    start = time.time()
    dequantized_weight_batch = fusion.dequantize_gptq_weight_batch_optimized(qweight, qzeros, scales, groupsize)
    batch_time = time.time() - start
    
    # 比较结果
    diff = torch.abs(dequantized_weight - dequantized_weight_batch).max()
    speedup = vectorized_time / batch_time if batch_time > 0 else float('inf')
    
    print(f"📊 最大差异: {diff.item():.6f}")
    print(f"🚀 批量优化加速比: {speedup:.2f}x")
    
    if diff < 1e-6:
        print("🎉 两种方法结果完全一致！")
    else:
        print("⚠️  结果有微小差异")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        sys.exit(1)
    
    test_optimization()
    test_small_matrix()
