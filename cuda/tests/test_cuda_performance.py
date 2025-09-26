#!/usr/bin/env python3
"""
CUDA内核性能测试
"""

import torch
import time
import sys
import os

def test_cuda_performance():
    """测试CUDA内核性能"""
    print("🚀 CUDA内核性能测试开始...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 导入编译的CUDA内核
    try:
        import fused_gptq_gemm_cuda
        print("✅ CUDA内核导入成功!")
    except ImportError:
        print("❌ CUDA内核未编译，请先运行编译脚本")
        return
    
    # 测试Qwen7B QKV投影
    print("\n🧪 测试Qwen7B QKV投影...")
    test_qkv_projection(fused_gptq_gemm_cuda)
    
    # 测试Qwen7B输出投影
    print("\n🧪 测试Qwen7B输出投影...")
    test_output_projection(fused_gptq_gemm_cuda)

def test_qkv_projection(fused_gptq_gemm_cuda):
    """测试QKV投影性能"""
    # Qwen7B QKV投影参数
    M, K, N = 1, 4096, 12288
    groupsize = 128
    num_groups = K // groupsize
    
    # 生成测试数据
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.bfloat16, device='cuda')
    
    print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}, qzeros{num_groups}x{K//8}, scales{num_groups}x{K}")
    
    # 预热
    print("🔥 预热CUDA内核...")
    for _ in range(10):
        _ = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
    torch.cuda.synchronize()
    
    # 性能测试
    print("⚡ 性能测试...")
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        result = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time)
        times.append(elapsed)
        
        if i % 20 == 0:
            print(f"  迭代 {i}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Qwen7B QKV投影性能统计:")
    print(f"  平均时间: {avg_time:.2f}ms")
    print(f"  最小时间: {min_time:.2f}ms")
    print(f"  最大时间: {max_time:.2f}ms")
    print(f"  结果形状: {result.shape}")
    
    # 性能评估
    target_time = 0.1  # 目标0.1ms
    if avg_time < target_time:
        print(f"🎉 性能优秀! 达到目标 {target_time}ms")
    elif avg_time < target_time * 2:
        print(f"✅ 性能良好，接近目标 {target_time}ms")
    else:
        print(f"⚠️ 性能需要优化，目标 {target_time}ms")

def test_output_projection(fused_gptq_gemm_cuda):
    """测试输出投影性能"""
    # Qwen7B输出投影参数
    M, K, N = 1, 4096, 4096
    groupsize = 128
    num_groups = K // groupsize
    
    # 生成测试数据
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.bfloat16, device='cuda')
    
    print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}, qzeros{num_groups}x{K//8}, scales{num_groups}x{K}")
    
    # 预热
    print("🔥 预热CUDA内核...")
    for _ in range(10):
        _ = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
    torch.cuda.synchronize()
    
    # 性能测试
    print("⚡ 性能测试...")
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        result = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time)
        times.append(elapsed)
        
        if i % 20 == 0:
            print(f"  迭代 {i}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Qwen7B输出投影性能统计:")
    print(f"  平均时间: {avg_time:.2f}ms")
    print(f"  最小时间: {min_time:.2f}ms")
    print(f"  最大时间: {max_time:.2f}ms")
    print(f"  结果形状: {result.shape}")
    
    # 性能评估
    target_time = 0.2  # 目标0.2ms
    if avg_time < target_time:
        print(f"🎉 性能优秀! 达到目标 {target_time}ms")
    elif avg_time < target_time * 2:
        print(f"✅ 性能良好，接近目标 {target_time}ms")
    else:
        print(f"⚠️ 性能需要优化，目标 {target_time}ms")

if __name__ == "__main__":
    test_cuda_performance()
