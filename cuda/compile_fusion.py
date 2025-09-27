#!/usr/bin/env python3
"""
编译GPTQ LN+QKV融合内核
"""

import torch
import os
import sys
from torch.utils.cpp_extension import load

def compile_fusion_kernel():
    """编译融合内核"""
    try:
        # 获取CUDA版本
        cuda_version = torch.version.cuda
        print(f"检测到CUDA版本: {cuda_version}")
        
        # 编译融合内核
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=True
        )
        
        print("✅ GPTQ LN+QKV融合内核编译成功!")
        return kernel_module
        
    except Exception as e:
        print(f"❌ 编译失败: {e}")
        return None

def test_compile():
    """测试编译结果"""
    try:
        # 创建测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 512
        groupsize = 128
        eps = 1e-5
        
        # 创建随机输入
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
        
        # 创建GPTQ参数（分别创建Q、K、V）
        qweight_q = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                dtype=torch.uint32, device='cuda')
        qweight_k = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                dtype=torch.uint32, device='cuda')
        qweight_v = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                dtype=torch.uint32, device='cuda')
        
        qzeros_q = torch.randint(0, 15, (8, 16), 
                               dtype=torch.uint32, device='cuda')
        qzeros_k = torch.randint(0, 15, (8, 16), 
                               dtype=torch.uint32, device='cuda')
        qzeros_v = torch.randint(0, 15, (8, 16), 
                               dtype=torch.uint32, device='cuda')
        
        scales_q = torch.randn(8, hidden_dim, 
                             dtype=torch.float16, device='cuda')
        scales_k = torch.randn(8, hidden_dim, 
                             dtype=torch.float16, device='cuda')
        scales_v = torch.randn(8, hidden_dim, 
                             dtype=torch.float16, device='cuda')
        
        # 创建LayerNorm参数
        ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
        
        print("🧪 进行简单功能测试...")
        
        # 调用融合内核
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight_q, qweight_k, qweight_v,
            qzeros_q, qzeros_k, qzeros_v,
            scales_q, scales_k, scales_v,
            ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
        
        # 解包QKV
        q_output = qkv_output[0]
        k_output = qkv_output[1] 
        v_output = qkv_output[2]
        
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        
        # 检查输出是否全零
        if torch.allclose(q_output, torch.zeros_like(q_output)) and \
           torch.allclose(k_output, torch.zeros_like(k_output)) and \
           torch.allclose(v_output, torch.zeros_like(v_output)):
            print("❌ 测试失败: 输出全零")
            return False
        
        print("✅ 功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 GPTQ LN+QKV融合内核编译开始...")
    
    # 编译内核
    kernel_module = compile_fusion_kernel()
    if kernel_module is None:
        print("❌ 编译失败")
        return
    
    # 测试编译结果
    if test_compile():
        print("✅ 编译和测试成功!")
    else:
        print("❌ 测试失败")

if __name__ == "__main__":
    main()