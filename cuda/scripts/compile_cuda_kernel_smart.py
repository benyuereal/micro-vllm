#!/usr/bin/env python3
"""
CUDA内核编译脚本 - 智能路径检测版本
"""

import os
import torch
import sys
from torch.utils.cpp_extension import load

def find_cuda_file():
    """智能查找CUDA文件"""
    possible_paths = [
        "src/gptq_cuda_kernel.cu",           # 从cuda目录运行
        "../src/gptq_cuda_kernel.cu",        # 从scripts目录运行
        "../../src/gptq_cuda_kernel.cu",     # 从tests目录运行
        "core/layer/gptq_cuda_kernel.cu",    # 从根目录运行
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def compile_cuda_kernel():
    """编译CUDA C++内核"""
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法编译CUDA内核")
    
    # 获取CUDA版本
    cuda_version = torch.version.cuda
    print(f"检测到CUDA版本: {cuda_version}")
    
    # 智能查找CUDA文件
    cuda_file = find_cuda_file()
    if cuda_file is None:
        print("❌ 找不到CUDA文件，请检查文件位置")
        print("📋 查找的路径:")
        print("  - src/gptq_cuda_kernel.cu")
        print("  - ../src/gptq_cuda_kernel.cu")
        print("  - ../../src/gptq_cuda_kernel.cu")
        print("  - core/layer/gptq_cuda_kernel.cu")
        return None
    
    print(f"✅ CUDA文件存在: {cuda_file}")
    
    # 编译选项
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-lineinfo",
        "-Xptxas=-O3",
        "--ptxas-options=-v",
        "-maxrregcount=255"
    ]
    
    # 编译内核
    try:
        print("🔨 开始编译CUDA内核...")
        fused_gptq_gemm = load(
            name="fused_gptq_gemm_cuda",
            sources=[cuda_file],
            extra_cuda_cflags=extra_cuda_cflags,
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
        print("⚠️ 编译失败，将使用Triton内核")
