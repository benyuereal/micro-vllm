#!/usr/bin/env python3
"""
CUDA内核测试 - 简化版本
"""

import torch
import time
import os
import sys
from torch.utils.cpp_extension import load

def test_cuda_kernel():
    """测试CUDA内核性能"""
    print("🚀 CUDA内核测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 直接编译CUDA内核
    print("\n🔨 编译CUDA内核...")
    try:
        fused_gptq_gemm_cuda = load(
            name="fused_gptq_gemm_cuda",
            sources=["gptq_cuda_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math", "-Xptxas=-O3"],
            verbose=False
        )
        print("✅ CUDA内核编译成功!")
    except Exception as e:
        print(f"❌ CUDA内核编译失败: {e}")
        return
    
    # 测试数据
    M, K, N = 1, 4096, 12288
    groupsize = 128
    num_groups = K // groupsize
    
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}")
    
    # 功能测试
    print("\n🧪 功能测试...")
    try:
        output = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        print("✅ 功能测试成功!")
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 输出数据类型: {output.dtype}")
        assert output.shape == (M, N)
        assert output.dtype == torch.float16
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return
    
    # 性能测试
    print("\n⚡ 性能测试...")
    num_warmup = 5
    num_runs = 20
    
    # 预热
    for _ in range(num_warmup):
        fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
    torch.cuda.synchronize()
    
    timings = []
    for i in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
        
        if i % 5 == 0:
            print(f"  迭代 {i}: {timings[-1]:.2f}ms")
    
    avg_time = sum(timings) / num_runs
    min_time = min(timings)
    max_time = max(timings)
    
    print(f"\n📊 性能统计:")
    print(f"  平均时间: {avg_time:.2f}ms")
    print(f"  最小时间: {min_time:.2f}ms")
    print(f"  最大时间: {max_time:.2f}ms")
    
    # 目标评估
    target_time = 0.10
    if avg_time < target_time:
        print(f"🎉 性能优秀! 达到目标 {target_time}ms")
    elif avg_time < target_time * 2:
        print(f"✅ 性能良好，接近目标 {target_time}ms")
    elif avg_time < target_time * 5:
        print(f"⚠️ 性能需要优化，目标 {target_time}ms")
    else:
        print(f"❌ 性能不达标，目标 {target_time}ms")
    
    print("\n🎉 CUDA内核测试完成!")

if __name__ == "__main__":
    test_cuda_kernel()
