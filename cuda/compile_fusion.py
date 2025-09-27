#!/usr/bin/env python3
"""
简化的融合LN+QKV GPTQ CUDA内核编译脚本
"""

import os
import torch
from torch.utils.cpp_extension import load

def compile_fusion_kernel():
    """编译融合内核"""
    print("🚀 融合LN+QKV GPTQ CUDA内核编译开始...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法编译CUDA内核")
    
    cuda_version = torch.version.cuda
    print(f"检测到CUDA版本: {cuda_version}")
    
    # 检查源文件
    source_file = "ln_qkv_fusion_kernel.cu"
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件 {source_file} 不存在")
    
    print(f"源文件: {source_file}")
    
    try:
        # 编译融合内核
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=[source_file],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=True
        )
        
        print("✅ 融合内核编译成功!")
        return kernel_module
        
    except Exception as e:
        print(f"❌ 编译失败: {e}")
        raise

def test_compile():
    """测试编译"""
    try:
        kernel_module = compile_fusion_kernel()
        
        # 简单测试
        print("🧪 进行简单功能测试...")
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 512
        num_heads, kv_num_heads, head_size = 8, 8, 64
        groupsize = 128
        eps = 1e-5
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # 输出张量
        q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        
        # 调用内核
        kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            q_output, k_output, v_output,
            batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
        )
        
        print("✅ 简单功能测试成功!")
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        success = test_compile()
        if success:
            print("🎉 编译和测试完成!")
        else:
            print("❌ 编译或测试失败")
        return success
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
