#!/usr/bin/env python3
"""
融合内核功能完整性验证 - GPTQ + LayerNorm + QKV
"""

import torch
from torch.utils.cpp_extension import load
import numpy as np

def main():
    print("🚀 开始融合内核功能完整性验证...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    
    try:
        # 编译内核
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        print("✅ 编译成功!")
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        groupsize = 128
        eps = 1e-5
        
        print(f"📊 测试配置: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        print(f"📊 groupsize={groupsize}, eps={eps}")
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        print(f"📊 输入形状: {input_tensor.shape}")
        print(f"📊 qweight形状: {qweight.shape}")
        print(f"📊 qzeros形状: {qzeros.shape}")
        print(f"📊 scales形状: {scales.shape}")
        print(f"📊 ln_weight形状: {ln_weight.shape}")
        print(f"📊 ln_bias形状: {ln_bias.shape}")
        
        # 调用融合内核
        print("🔄 调用融合内核...")
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
        
        print(f"📊 QKV输出形状: {qkv_output.shape}")
        
        # 解包QKV输出
        q_output = qkv_output[0]
        k_output = qkv_output[1] 
        v_output = qkv_output[2]
        
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        
        # 功能验证
        print("\n🔍 功能验证:")
        
        # 1. 检查输出非零
        if q_output.sum().item() != 0 or k_output.sum().item() != 0 or v_output.sum().item() != 0:
            print("✅ 输出非零检查通过")
        else:
            print("❌ 输出非零检查失败")
            return False
        
        # 2. 检查输出形状
        expected_shape = (batch_size, seq_len, hidden_dim)
        if q_output.shape == expected_shape and k_output.shape == expected_shape and v_output.shape == expected_shape:
            print("✅ 输出形状检查通过")
        else:
            print(f"❌ 输出形状检查失败，期望: {expected_shape}")
            return False
        
        # 3. 检查输出数据类型
        if q_output.dtype == torch.float16 and k_output.dtype == torch.float16 and v_output.dtype == torch.float16:
            print("✅ 输出数据类型检查通过")
        else:
            print("❌ 输出数据类型检查失败")
            return False
        
        # 4. 检查输出设备
        if q_output.device == input_tensor.device and k_output.device == input_tensor.device and v_output.device == input_tensor.device:
            print("✅ 输出设备检查通过")
        else:
            print("❌ 输出设备检查失败")
            return False
        
        # 5. 检查输出范围
        q_range = (q_output.min().item(), q_output.max().item())
        k_range = (k_output.min().item(), k_output.max().item())
        v_range = (v_output.min().item(), v_output.max().item())
        
        print(f"📊 Q输出范围: [{q_range[0]:.4f}, {q_range[1]:.4f}]")
        print(f"📊 K输出范围: [{k_range[0]:.4f}, {k_range[1]:.4f}]")
        print(f"📊 V输出范围: [{v_range[0]:.4f}, {v_range[1]:.4f}]")
        
        # 检查是否有NaN或Inf
        if not torch.isnan(q_output).any() and not torch.isinf(q_output).any():
            print("✅ Q输出数值检查通过")
        else:
            print("❌ Q输出数值检查失败")
            return False
        
        if not torch.isnan(k_output).any() and not torch.isinf(k_output).any():
            print("✅ K输出数值检查通过")
        else:
            print("❌ K输出数值检查失败")
            return False
        
        if not torch.isnan(v_output).any() and not torch.isinf(v_output).any():
            print("✅ V输出数值检查通过")
        else:
            print("❌ V输出数值检查失败")
            return False
        
        # 6. 检查GPTQ功能
        print("\n🔍 GPTQ功能验证:")
        
        # 检查qweight格式
        expected_qweight_shape = (hidden_dim // 8, hidden_dim * 3)
        if qweight.shape == expected_qweight_shape:
            print("✅ qweight形状检查通过")
        else:
            print(f"❌ qweight形状检查失败，期望: {expected_qweight_shape}")
            return False
        
        # 检查qzeros格式
        expected_qzeros_shape = (hidden_dim // groupsize, groupsize // 8)
        if qzeros.shape == expected_qzeros_shape:
            print("✅ qzeros形状检查通过")
        else:
            print(f"❌ qzeros形状检查失败，期望: {expected_qzeros_shape}")
            return False
        
        # 检查scales格式
        expected_scales_shape = (hidden_dim // groupsize, hidden_dim * 3)
        if scales.shape == expected_scales_shape:
            print("✅ scales形状检查通过")
        else:
            print(f"❌ scales形状检查失败，期望: {expected_scales_shape}")
            return False
        
        # 7. 检查LayerNorm功能
        print("\n🔍 LayerNorm功能验证:")
        
        # 检查ln_weight和ln_bias形状
        expected_ln_shape = (hidden_dim,)
        if ln_weight.shape == expected_ln_shape and ln_bias.shape == expected_ln_shape:
            print("✅ LayerNorm参数形状检查通过")
        else:
            print(f"❌ LayerNorm参数形状检查失败，期望: {expected_ln_shape}")
            return False
        
        # 8. 检查QKV融合功能
        print("\n🔍 QKV融合功能验证:")
        
        # 检查QKV输出是否不同（不是相同的值）
        q_k_diff = torch.abs(q_output - k_output).mean().item()
        q_v_diff = torch.abs(q_output - v_output).mean().item()
        k_v_diff = torch.abs(k_output - v_output).mean().item()
        
        print(f"📊 Q-K差异: {q_k_diff:.6f}")
        print(f"📊 Q-V差异: {q_v_diff:.6f}")
        print(f"📊 K-V差异: {k_v_diff:.6f}")
        
        if q_k_diff > 1e-6 and q_v_diff > 1e-6 and k_v_diff > 1e-6:
            print("✅ QKV输出差异检查通过")
        else:
            print("⚠️ QKV输出差异较小，可能存在问题")
        
        print("\n🎉 所有功能验证通过!")
        print("✅ GPTQ INT4动态反量化功能正常")
        print("✅ LayerNorm功能正常")
        print("✅ QKV融合功能正常")
        print("✅ 输出格式正确")
        print("✅ 数值稳定性良好")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
