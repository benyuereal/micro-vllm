#!/usr/bin/env python3
"""
只测试LayerNorm部分的精度
"""

import torch
import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_layernorm_only():
    """只测试LayerNorm部分，不包含GPTQ投影"""
    try:
        logger.info("🔧 编译融合内核...")
        
        # 编译融合内核
        from torch.utils.cpp_extension import load
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=[os.path.join(os.path.dirname(__file__), '..', '..', 'cuda', 'gptq_ln_qkv_fusion_kernel.cu')],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        logger.info("✅ 融合内核加载成功!")
        
        # 测试参数
        batch_size, seq_len, hidden_dim = 1, 1, 512
        groupsize = 128
        eps = 1e-5
        
        # 创建测试数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
        
        # 创建单位矩阵的GPTQ参数（这样GPTQ投影就是恒等变换）
        # 对于4bit量化，我们需要创建特殊的权重矩阵
        qweight_q = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
        qweight_k = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
        qweight_v = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
        
        # 创建单位矩阵的4bit表示
        # 对于4bit，值8表示1.0（因为 (8-8)*1.0 = 0，但我们需要1.0）
        # 让我们使用值8来表示1.0
        for i in range(hidden_dim):
            byte_idx = i // 8
            bit_idx = i % 8
            # 设置值为8（在4bit中表示0，但通过scales调整）
            qweight_q[byte_idx, i] = 8 << (bit_idx * 4)
            qweight_k[byte_idx, i] = 8 << (bit_idx * 4)
            qweight_v[byte_idx, i] = 8 << (bit_idx * 4)
        
        qzeros_q = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
        qzeros_k = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
        qzeros_v = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
        
        # 设置scales为单位矩阵
        scales_q = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
        scales_k = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
        scales_v = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
        
        logger.info("🧪 测试LayerNorm精度...")
        
        # 调用融合内核
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight_q, qweight_k, qweight_v,
            qzeros_q, qzeros_k, qzeros_v,
            scales_q, scales_k, scales_v,
            ln_weight, ln_bias,
            batch_size, seq_len, hidden_dim, groupsize, eps
        )
        
        # 解包QKV输出
        q_output = qkv_output[0]
        k_output = qkv_output[1]
        v_output = qkv_output[2]
        
        # 计算PyTorch LayerNorm作为参考
        pytorch_ln = torch.nn.LayerNorm(hidden_dim, eps=eps, dtype=torch.float16, device='cuda')
        pytorch_ln.weight.data = ln_weight
        pytorch_ln.bias.data = ln_bias
        
        pytorch_output = pytorch_ln(input_tensor)
        
        # 计算差异
        q_diff = torch.abs(q_output - pytorch_output).max().item()
        k_diff = torch.abs(k_output - pytorch_output).max().item()
        v_diff = torch.abs(v_output - pytorch_output).max().item()
        
        logger.info("📊 LayerNorm精度测试结果:")
        logger.info(f"  📊 Q输出差异: {q_diff:.6f}")
        logger.info(f"  📊 K输出差异: {k_diff:.6f}")
        logger.info(f"  📊 V输出差异: {v_diff:.6f}")
        logger.info(f"  📊 PyTorch输出范围: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
        logger.info(f"  📊 Q输出范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
        
        # 检查精度
        tolerance = 1e-3
        if q_diff < tolerance and k_diff < tolerance and v_diff < tolerance:
            logger.info("✅ LayerNorm精度测试通过!")
            return True
        else:
            logger.warning(f"⚠️ LayerNorm精度未达标: 最大差异 {max(q_diff, k_diff, v_diff):.6f} > {tolerance}")
            return False
            
    except Exception as e:
        logger.error(f"❌ LayerNorm测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_layernorm_only()
    if success:
        logger.info("🎉 LayerNorm测试通过!")
    else:
        logger.error("❌ LayerNorm测试失败!")
        sys.exit(1)
