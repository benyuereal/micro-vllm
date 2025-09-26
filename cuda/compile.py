#!/usr/bin/env python3
"""
CUDA内核编译脚本
"""

import os
import torch
from torch.utils.cpp_extension import load

def compile_cuda_kernel():
    """编译CUDA C++内核"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法编译CUDA内核")
    
    cuda_version = torch.version.cuda
    print(f"检测到CUDA版本: {cuda_version}")
    
    cuda_file = "gptq_cuda_kernel.cu"
    if not os.path.exists(cuda_file):
        print(f"❌ CUDA文件不存在: {cuda_file}")
        return None
    
    print(f"✅ CUDA文件存在: {cuda_file}")
    
    try:
        print("🔨 开始编译CUDA内核...")
        fused_gptq_gemm = load(
            name="fused_gptq_gemm_cuda",
            sources=[cuda_file],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=True
        )
        print("✅ CUDA C++内核编译成功!")
        return fused_gptq_gemm
    except Exception as e:
        print(f"❌ CUDA C++内核编译失败: {e}")
        return None

if __name__ == "__main__":
    kernel = compile_cuda_kernel()
    if kernel:
        print("🎉 CUDA内核已准备就绪!")
    else:
        print("⚠️ 编译失败")
