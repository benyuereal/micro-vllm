#!/usr/bin/env python3
"""
融合LN+QKV GPTQ快速测试脚本
用于快速验证功能
"""

import sys
import os
import torch
import time
import logging
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """快速测试"""
    logger.info("🚀 开始快速测试...")
    
    # 检查CUDA环境
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行测试")
        return False
    
    logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
    
    try:
        # 导入融合算子
        from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
        
        # 创建测试数据
        input_data = torch.randn(1, 1, 4096, dtype=torch.float16)
        
        # 创建模拟的GPTQ参数
        qweight = torch.randint(0, 255, (512, 12288), dtype=torch.uint32)
        qzeros = torch.randint(0, 255, (32, 1536), dtype=torch.uint32)
        scales = torch.randn(32, 12288, dtype=torch.float16)
        
        # 创建LayerNorm参数
        ln_weight = torch.randn(4096, dtype=torch.float16)
        ln_bias = torch.randn(4096, dtype=torch.float16)
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用")
            return False
        
        # 测试融合算子
        logger.info("🧪 测试融合算子...")
        start_time = time.time()
        
        q, k, v = fused_op.fused_ln_qkv_gptq(
            input=input_data,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            ln_weight=ln_weight,
            ln_bias=ln_bias,
            num_heads=32,
            kv_num_heads=32,
            head_size=128,
            groupsize=128
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        
        # 验证输出形状
        expected_shape = (1, 32, 1, 128)
        assert q.shape == expected_shape, f"Q形状错误: {q.shape} != {expected_shape}"
        assert k.shape == expected_shape, f"K形状错误: {k.shape} != {expected_shape}"
        assert v.shape == expected_shape, f"V形状错误: {v.shape} != {expected_shape}"
        
        # 检查输出是否包含有效值
        if torch.any(torch.isnan(q)) or torch.any(torch.isinf(q)):
            logger.error("❌ Q输出包含无效值")
            return False
        
        if torch.any(torch.isnan(k)) or torch.any(torch.isinf(k)):
            logger.error("❌ K输出包含无效值")
            return False
        
        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
            logger.error("❌ V输出包含无效值")
            return False
        
        logger.info(f"✅ 快速测试通过!")
        logger.info(f"📊 执行时间: {elapsed_time:.2f}ms")
        logger.info(f"📊 输出形状: Q{q.shape}, K{k.shape}, V{v.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 快速测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
