#!/usr/bin/env python3
"""
最简单的编译测试
"""

import torch
from torch.utils.cpp_extension import load

def main():
    print("🚀 开始编译测试...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    
    try:
        # 编译内核
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        print("✅ 编译成功!")
        
        # 简单测试
        print("🧪 进行简单测试...")
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 256
        num_heads, kv_num_heads, head_size = 4, 4, 64
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
        
        print("✅ 测试成功!")
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
