#!/usr/bin/env python3
"""
vLLM版本CUDA内核编译脚本
"""

import os
import torch
from torch.utils.cpp_extension import load

def compile_vllm_kernel():
    """编译vLLM版本CUDA内核"""
    print("🚀 vLLM版本CUDA内核编译开始...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法编译CUDA内核")
    
    cuda_version = torch.version.cuda
    print(f"检测到CUDA版本: {cuda_version}")
    
    # vLLM版本文件
    vllm_file = "gptq_cuda_kernel_vllm.cu"
    if not os.path.exists(vllm_file):
        print(f"❌ vLLM CUDA文件不存在: {vllm_file}")
        return None
    
    print(f"✅ vLLM CUDA文件存在: {vllm_file}")
    
    # 优化编译选项
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-Xptxas=-O3",
        "--ptxas-options=-v",
        "-maxrregcount=255",
        "-lineinfo"
    ]
    
    # 编译vLLM内核
    try:
        print("🔨 开始编译vLLM CUDA内核...")
        fused_gptq_gemm_vllm = load(
            name="fused_gptq_gemm_cuda_vllm",
            sources=[vllm_file],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False
        )
        print("✅ vLLM CUDA内核编译成功!")
        return fused_gptq_gemm_vllm
    except Exception as e:
        print(f"❌ vLLM CUDA内核编译失败: {e}")
        return None

def test_vllm_kernel():
    """测试vLLM内核功能"""
    print("\n🧪 测试vLLM内核功能...")
    
    try:
        # 导入编译的内核
        import fused_gptq_gemm_cuda_vllm
        
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
        output = fused_gptq_gemm_cuda_vllm.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        
        print("✅ vLLM内核功能测试成功!")
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 输出数据类型: {output.dtype}")
        
        assert output.shape == (M, N)
        assert output.dtype == torch.float16
        
        return True
        
    except Exception as e:
        print(f"❌ vLLM内核功能测试失败: {e}")
        return False

def benchmark_vllm_kernel():
    """vLLM内核性能测试"""
    print("\n⚡ vLLM内核性能测试...")
    
    try:
        import fused_gptq_gemm_cuda_vllm
        
        # 测试数据
        M, K, N = 1, 4096, 12288
        groupsize = 128
        num_groups = K // groupsize
        
        input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        # 预热
        for _ in range(10):
            fused_gptq_gemm_cuda_vllm.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        torch.cuda.synchronize()
        
        # 性能测试
        num_runs = 50
        timings = []
        
        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            fused_gptq_gemm_cuda_vllm.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
            if i % 10 == 0:
                print(f"  迭代 {i}: {timings[-1]:.2f}ms")
        
        avg_time = sum(timings) / num_runs
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"\n📊 vLLM内核性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        
        # 性能评估
        target_time = 0.10
        if avg_time < target_time:
            print("🎉 vLLM内核达到目标性能!")
        elif avg_time < target_time * 2:
            print("✅ vLLM内核接近目标性能")
        elif avg_time < target_time * 5:
            print("⚠️ vLLM内核需要进一步优化")
        else:
            print("❌ vLLM内核性能不达标")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ vLLM内核性能测试失败: {e}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 vLLM版本CUDA内核编译和测试")
    print("=" * 50)
    
    # 编译内核
    kernel = compile_vllm_kernel()
    if kernel is None:
        print("❌ 编译失败，退出")
        exit(1)
    
    # 功能测试
    if not test_vllm_kernel():
        print("❌ 功能测试失败，退出")
        exit(1)
    
    # 性能测试
    avg_time = benchmark_vllm_kernel()
    if avg_time is None:
        print("❌ 性能测试失败，退出")
        exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 vLLM版本CUDA内核测试完成!")
    print(f"📊 最终性能: {avg_time:.2f}ms")
    print("=" * 50)
