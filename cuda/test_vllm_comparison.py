#!/usr/bin/env python3
"""
CUDA内核性能对比测试 - vLLM风格 vs 当前版本
"""

import torch
import time
import os
import sys
from torch.utils.cpp_extension import load

def test_performance_comparison():
    """性能对比测试"""
    print("🚀 CUDA内核性能对比测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 测试数据
    M, K, N = 1, 4096, 12288
    groupsize = 128
    num_groups = K // groupsize
    
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}")
    
    # 测试当前版本
    print("\n⚡ 测试当前版本")
    print("==========================================")
    try:
        # 导入当前版本
        try:
            import fused_gptq_gemm_cuda
            print("✅ 当前版本导入成功!")
        except ImportError:
            print("🔨 编译当前版本...")
            fused_gptq_gemm_cuda = load(
                name="fused_gptq_gemm_cuda",
                sources=["gptq_cuda_kernel.cu"],
                extra_cuda_cflags=["-O3", "-use_fast_math", "-Xptxas=-O3"],
                verbose=False
            )
            print("✅ 当前版本编译成功!")
        
        # 测试当前版本性能
        times_current = []
        for i in range(20):
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
            times_current.append(elapsed)
            
            if i % 5 == 0:
                print(f"  迭代 {i}: {elapsed:.2f}ms")
        
        avg_current = sum(times_current) / len(times_current)
        min_current = min(times_current)
        max_current = max(times_current)
        
        print(f"📊 当前版本性能统计:")
        print(f"  平均时间: {avg_current:.2f}ms")
        print(f"  最小时间: {min_current:.2f}ms")
        print(f"  最大时间: {max_current:.2f}ms")
        
    except Exception as e:
        print(f"❌ 当前版本测试失败: {e}")
        return
    
    # 测试vLLM风格版本
    print("\n⚡ 测试vLLM风格版本")
    print("==========================================")
    try:
        print("🔨 编译vLLM风格版本...")
        vllm_kernel = load(
            name="fused_gptq_gemm_cuda_vllm",
            sources=["gptq_cuda_kernel_vllm.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math", "-Xptxas=-O3", "-lcublas"],
            verbose=False
        )
        print("✅ vLLM风格版本编译成功!")
        
        # 测试vLLM风格版本性能
        times_vllm = []
        for i in range(20):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = vllm_kernel.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            times_vllm.append(elapsed)
            
            if i % 5 == 0:
                print(f"  迭代 {i}: {elapsed:.2f}ms")
        
        avg_vllm = sum(times_vllm) / len(times_vllm)
        min_vllm = min(times_vllm)
        max_vllm = max(times_vllm)
        
        print(f"📊 vLLM风格版本性能统计:")
        print(f"  平均时间: {avg_vllm:.2f}ms")
        print(f"  最小时间: {min_vllm:.2f}ms")
        print(f"  最大时间: {max_vllm:.2f}ms")
        
    except Exception as e:
        print(f"❌ vLLM风格版本测试失败: {e}")
        avg_vllm = float('inf')
    
    # 性能对比
    print(f"\n📊 性能对比结果:")
    print(f"  当前版本: {avg_current:.2f}ms")
    if avg_vllm != float('inf'):
        print(f"  vLLM风格版本: {avg_vllm:.2f}ms")
        print(f"  性能提升: {avg_current/avg_vllm:.2f}x")
    
    # 目标评估
    target_time = 0.10
    best_time = min(avg_current, avg_vllm if avg_vllm != float('inf') else float('inf'))
    
    if best_time < target_time:
        print(f"🎉 最佳版本达到目标 {target_time}ms!")
    elif best_time < target_time * 2:
        print(f"✅ 最佳版本接近目标 {target_time}ms")
    elif best_time < target_time * 5:
        print(f"⚠️ 最佳版本需要优化，目标 {target_time}ms")
    else:
        print(f"❌ 所有版本性能不达标，目标 {target_time}ms")
    
    print("\n🎉 性能对比测试完成!")

if __name__ == "__main__":
    test_performance_comparison()
