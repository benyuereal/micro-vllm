#!/usr/bin/env python3
"""
测试大矩阵上的GPTQ性能
"""
import torch
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def test_large_matrix():
    """测试大矩阵上的性能"""
    print("🔥 大矩阵GPTQ性能测试...")
    
    # 使用更大的矩阵尺寸
    M, K, N = 1024, 4096, 2048  # 大矩阵
    groupsize = 128
    
    # 创建GPTQ格式的张量
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    num_groups = K // groupsize
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"qweight形状: {qweight.shape}")
    print(f"scales形状: {scales.shape}")
    print(f"qzeros形状: {qzeros.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 预热
    print("🔥 预热Triton内核...")
    for _ in range(3):
        _ = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
    torch.cuda.synchronize()
    print("✅ 预热完成")
    
    try:
        # 测试融合算子（多次运行取平均）
        print("🚀 测试融合算子...")
        fusion_times = []
        for i in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            result = fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
            torch.cuda.synchronize()
            fusion_times.append(time.time() - start_time)
        
        avg_fusion_time = sum(fusion_times) / len(fusion_times)
        print(f"✅ 融合算子平均耗时: {avg_fusion_time:.4f}s")
        
        # 测试基线实现（多次运行取平均）
        print("🐌 测试基线实现...")
        baseline_times = []
        for i in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            baseline_result = fusion.baseline_gptq_gemm(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                groupsize=groupsize
            )
            torch.cuda.synchronize()
            baseline_times.append(time.time() - start_time)
        
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        print(f"✅ 基线实现平均耗时: {avg_baseline_time:.4f}s")
        
        # 比较结果
        diff = torch.abs(result - baseline_result).max()
        speedup = avg_baseline_time / avg_fusion_time
        
        print(f"📊 最大差异: {diff.item():.6f}")
        print(f"🚀 加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("🎉 GPTQ融合算子在大矩阵上表现更好！")
        else:
            print("⚠️  基线实现仍然更快，可能需要进一步优化Triton内核")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_medium_matrix():
    """测试中等矩阵上的性能"""
    print("\n🔥 中等矩阵GPTQ性能测试...")
    
    # 使用中等矩阵尺寸
    M, K, N = 256, 1024, 512
    groupsize = 128
    
    # 创建GPTQ格式的张量
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    num_groups = K // groupsize
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    
    print(f"输入形状: {input_tensor.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 预热
    print("🔥 预热Triton内核...")
    for _ in range(3):
        _ = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
    torch.cuda.synchronize()
    print("✅ 预热完成")
    
    try:
        # 测试融合算子
        print("🚀 测试融合算子...")
        torch.cuda.synchronize()
        start_time = time.time()
        result = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        torch.cuda.synchronize()
        fusion_time = time.time() - start_time
        print(f"✅ 融合算子耗时: {fusion_time:.4f}s")
        
        # 测试基线实现
        print("🐌 测试基线实现...")
        torch.cuda.synchronize()
        start_time = time.time()
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        torch.cuda.synchronize()
        baseline_time = time.time() - start_time
        print(f"✅ 基线实现耗时: {baseline_time:.4f}s")
        
        # 比较结果
        diff = torch.abs(result - baseline_result).max()
        speedup = baseline_time / fusion_time
        
        print(f"📊 最大差异: {diff.item():.6f}")
        print(f"🚀 加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("🎉 GPTQ融合算子在中型矩阵上表现更好！")
        else:
            print("⚠️  基线实现仍然更快")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        sys.exit(1)
    
    # 先测试中等矩阵
    test_medium_matrix()
    
    # 再测试大矩阵
    test_large_matrix()
