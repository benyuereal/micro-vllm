#!/usr/bin/env python3
"""
简单精度测试 - 验证融合内核的GPTQ解量化是否正确
"""

import sys
import os
import torch
import logging

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compile_fusion_kernel():
    """编译融合内核"""
    try:
        from torch.utils.cpp_extension import load
        
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["../../cuda/gptq_ln_qkv_fusion_kernel.cu"],
            verbose=False
        )
        logger.info("✅ 融合内核编译成功!")
        return kernel_module
    except Exception as e:
        logger.error(f"❌ 融合内核编译失败: {e}")
        return None

def test_simple_gptq():
    """测试简单的GPTQ解量化"""
    logger.info("🎯 开始简单GPTQ精度测试...")
    
    # 编译内核
    kernel_module = compile_fusion_kernel()
    if kernel_module is None:
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len, hidden_dim = 1, 1, 64  # 使用小尺寸便于调试
    groupsize = 32
    
    # 创建简单的输入数据
    input_data = torch.ones(batch_size, seq_len, hidden_dim, dtype=torch.float16, device=device)
    
    # 创建LayerNorm参数（单位变换）
    ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device=device)
    ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device=device)
    
    # 创建单位矩阵的GPTQ权重
    qweight_q = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device=device)
    qweight_k = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device=device)
    qweight_v = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device=device)
    
    # 设置单位矩阵：每个4bit值都是1
    for i in range(hidden_dim):
        byte_idx = i // 8
        bit_idx = i % 8
        shift = bit_idx * 4
        if shift < 32:
            qweight_q[byte_idx, i] = 1 << shift
            qweight_k[byte_idx, i] = 1 << shift
            qweight_v[byte_idx, i] = 1 << shift
    
    # 创建GPTQ参数
    num_groups = hidden_dim // groupsize
    qzeros_q = torch.zeros(num_groups, groupsize // 8, dtype=torch.uint32, device=device)
    qzeros_k = torch.zeros(num_groups, groupsize // 8, dtype=torch.uint32, device=device)
    qzeros_v = torch.zeros(num_groups, groupsize // 8, dtype=torch.uint32, device=device)
    
    scales_q = torch.ones(num_groups, hidden_dim, dtype=torch.float16, device=device)
    scales_k = torch.ones(num_groups, hidden_dim, dtype=torch.float16, device=device)
    scales_v = torch.ones(num_groups, hidden_dim, dtype=torch.float16, device=device)
    
    # 调用融合内核
    try:
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            input_data, ln_weight, ln_bias,
            qweight_q, qweight_k, qweight_v,
            qzeros_q, qzeros_k, qzeros_v,
            scales_q, scales_k, scales_v,
            batch_size, seq_len, hidden_dim, groupsize, 1e-5
        )
        
        q_output = qkv_output[0]
        k_output = qkv_output[1] 
        v_output = qkv_output[2]
        
        logger.info(f"📊 Q输出形状: {q_output.shape}")
        logger.info(f"📊 Q输出前5个值: {q_output[0, 0, :5]}")
        logger.info(f"📊 K输出前5个值: {k_output[0, 0, :5]}")
        logger.info(f"📊 V输出前5个值: {v_output[0, 0, :5]}")
        
        # 对于单位矩阵，输入全1，输出应该接近全1
        expected_value = 1.0  # 因为LayerNorm(1) = 1, 然后乘以单位矩阵 = 1
        
        q_diff = torch.abs(q_output - expected_value).max()
        k_diff = torch.abs(k_output - expected_value).max()
        v_diff = torch.abs(v_output - expected_value).max()
        
        logger.info(f"📊 Q与期望值差异: {q_diff:.6f}")
        logger.info(f"📊 K与期望值差异: {k_diff:.6f}")
        logger.info(f"📊 V与期望值差异: {v_diff:.6f}")
        
        # 检查是否接近期望值
        tolerance = 0.1  # 允许一定的数值误差
        if q_diff < tolerance and k_diff < tolerance and v_diff < tolerance:
            logger.info("✅ 简单GPTQ精度测试通过!")
            return True
        else:
            logger.warning(f"⚠️ 精度差异较大，可能存在问题")
            return False
            
    except Exception as e:
        logger.error(f"❌ 简单GPTQ精度测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始简单精度测试...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用")
        return
    
    logger.info(f"✅ CUDA可用，版本: {torch.version.cuda}")
    logger.info(f"📊 PyTorch版本: {torch.__version__}")
    
    success = test_simple_gptq()
    
    if success:
        logger.info("✅ 所有测试通过!")
    else:
        logger.error("❌ 部分测试失败!")

if __name__ == "__main__":
    main()
