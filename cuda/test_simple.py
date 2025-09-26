#!/usr/bin/env python3
"""
简化的CUDA内核测试脚本
"""

import torch
import time
import os
import sys

def test_cuda_kernel():
    """测试CUDA内核"""
    print("🚀 CUDA内核测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 直接编译并测试
    print("\n🔨 编译CUDA内核...")
    try:
        from torch.utils.cpp_extension import load
        
        cuda_file = "gptq_cuda_kernel.cu"
        if not os.path.exists(cuda_file):
            print(f"❌ CUDA文件不存在: {cuda_file}")
            return
        
        print(f"✅ CUDA文件存在: {cuda_file}")
        
        # 编译
        fused_gptq_gemm_cuda = load(
            name="fused_gptq_gemm_cuda",
            sources=[cuda_file],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
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
    
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.bfloat16, device='cuda')
    
    print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}")
    
    # 功能测试
    print("\n🧪 功能测试...")
    try:
        result = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        
        print(f"✅ 功能测试成功!")
        print(f"📊 输出形状: {result.shape}")
        print(f"📊 输出数据类型: {result.dtype}")
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return
    
    # 性能测试
    print("\n⚡ 性能测试...")
    try:
        # 预热
        for _ in range(10):
            _ = fused_gptq_gemm_cuda.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        torch.cuda.synchronize()
        
        # 测试
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
        
        print(f"\n📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        
        # 性能评估
        target_time = 0.1
        if avg_time < target_time:
            print(f"🎉 性能优秀! 达到目标 {target_time}ms")
        elif avg_time < target_time * 2:
            print(f"✅ 性能良好，接近目标 {target_time}ms")
        else:
            print(f"⚠️ 性能需要优化，目标 {target_time}ms")
            
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")

if __name__ == "__main__":
    test_cuda_kernel()
