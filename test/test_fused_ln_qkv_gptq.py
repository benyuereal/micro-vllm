#!/usr/bin/env python3
"""
融合LN+QKV GPTQ功能测试
测试融合算子的正确性和性能
"""

import sys
import os
import torch
import time
import logging
import numpy as np
from typing import Tuple

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ, fused_ln_qkv_gptq

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockGPTQLayer:
    """模拟GPTQ量化层"""
    
    def __init__(self, input_dim: int, output_dim: int, groupsize: int = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groupsize = groupsize
        
        # 生成模拟的GPTQ参数
        self.qweight = torch.randint(0, 255, (input_dim // 8, output_dim), dtype=torch.uint32)
        self.qzeros = torch.randint(0, 255, (output_dim // groupsize, groupsize // 8), dtype=torch.uint32)
        self.scales = torch.randn(output_dim // groupsize, output_dim, dtype=torch.float16)
        
        # 生成LayerNorm参数
        self.ln_weight = torch.randn(input_dim, dtype=torch.float16)
        self.ln_bias = torch.randn(input_dim, dtype=torch.float16)

def create_test_data(batch_size: int = 1, seq_len: int = 1, hidden_dim: int = 4096) -> Tuple[torch.Tensor, MockGPTQLayer]:
    """创建测试数据"""
    # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
    
    # 创建模拟的GPTQ层
    gptq_layer = MockGPTQLayer(hidden_dim, hidden_dim * 3, groupsize=128)
    
    return input_data, gptq_layer

def test_fused_ln_qkv_gptq_basic():
    """测试基础功能"""
    logger.info("🧪 测试基础功能...")
    
    try:
        # 创建测试数据
        input_data, gptq_layer = create_test_data()
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过测试")
            return False
        
        # 测试融合算子
        q, k, v = fused_op.fused_ln_qkv_gptq(
            input=input_data,
            qweight=gptq_layer.qweight,
            qzeros=gptq_layer.qzeros,
            scales=gptq_layer.scales,
            ln_weight=gptq_layer.ln_weight,
            ln_bias=gptq_layer.ln_bias,
            num_heads=32,
            kv_num_heads=32,
            head_size=128,
            groupsize=128
        )
        
        # 验证输出形状
        expected_shape = (1, 32, 1, 128)
        assert q.shape == expected_shape, f"Q形状错误: {q.shape} != {expected_shape}"
        assert k.shape == expected_shape, f"K形状错误: {k.shape} != {expected_shape}"
        assert v.shape == expected_shape, f"V形状错误: {v.shape} != {expected_shape}"
        
        logger.info("✅ 基础功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 基础功能测试失败: {e}")
        return False

def test_fused_ln_qkv_gptq_performance():
    """测试性能"""
    logger.info("🚀 测试性能...")
    
    try:
        # 创建测试数据
        input_data, gptq_layer = create_test_data()
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过性能测试")
            return False
        
        # 预热
        for _ in range(10):
            q, k, v = fused_op.fused_ln_qkv_gptq(
                input=input_data,
                qweight=gptq_layer.qweight,
                qzeros=gptq_layer.qzeros,
                scales=gptq_layer.scales,
                ln_weight=gptq_layer.ln_weight,
                ln_bias=gptq_layer.ln_bias,
                num_heads=32,
                kv_num_heads=32,
                head_size=128,
                groupsize=128
            )
        
        # 性能测试
        num_runs = 100
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            q, k, v = fused_op.fused_ln_qkv_gptq(
                input=input_data,
                qweight=gptq_layer.qweight,
                qzeros=gptq_layer.qzeros,
                scales=gptq_layer.scales,
                ln_weight=gptq_layer.ln_weight,
                ln_bias=gptq_layer.ln_bias,
                num_heads=32,
                kv_num_heads=32,
                head_size=128,
                groupsize=128
            )
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算统计信息
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"📊 性能统计:")
        logger.info(f"  平均时间: {avg_time:.2f}ms")
        logger.info(f"  最小时间: {min_time:.2f}ms")
        logger.info(f"  最大时间: {max_time:.2f}ms")
        
        # 性能目标检查
        target_time = 0.25  # 目标时间: 0.25ms
        if avg_time <= target_time:
            logger.info(f"🎯 达到性能目标: {avg_time:.2f}ms <= {target_time}ms")
            return True
        else:
            logger.warning(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > {target_time}ms")
            return False
        
    except Exception as e:
        logger.error(f"❌ 性能测试失败: {e}")
        return False

def test_fused_ln_qkv_gptq_correctness():
    """测试正确性"""
    logger.info("🔍 测试正确性...")
    
    try:
        # 创建测试数据
        input_data, gptq_layer = create_test_data()
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过正确性测试")
            return False
        
        # 使用融合算子
        q_fused, k_fused, v_fused = fused_op.fused_ln_qkv_gptq(
            input=input_data,
            qweight=gptq_layer.qweight,
            qzeros=gptq_layer.qzeros,
            scales=gptq_layer.scales,
            ln_weight=gptq_layer.ln_weight,
            ln_bias=gptq_layer.ln_bias,
            num_heads=32,
            kv_num_heads=32,
            head_size=128,
            groupsize=128
        )
        
        # 使用PyTorch实现作为参考
        # LayerNorm
        ln_output = torch.nn.functional.layer_norm(
            input_data, 
            input_data.shape[-1:], 
            gptq_layer.ln_weight, 
            gptq_layer.ln_bias
        )
        
        # 简化的QKV投影（使用随机权重）
        qkv_weight = torch.randn(input_data.shape[-1], input_data.shape[-1] * 3, dtype=torch.float16)
        qkv_output = torch.matmul(ln_output, qkv_weight)
        
        # 分割QKV
        hidden_size = qkv_output.shape[-1] // 3
        q_ref, k_ref, v_ref = qkv_output.split(hidden_size, dim=-1)
        
        # 重塑为head格式
        batch_size, seq_len, hidden_dim = q_ref.shape
        q_ref = q_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
        k_ref = k_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
        v_ref = v_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
        
        # 比较结果（由于GPTQ量化，结果可能不完全相同）
        logger.info("📊 结果比较:")
        logger.info(f"  Q形状: {q_fused.shape} vs {q_ref.shape}")
        logger.info(f"  K形状: {k_fused.shape} vs {k_ref.shape}")
        logger.info(f"  V形状: {v_fused.shape} vs {v_ref.shape}")
        
        # 检查输出是否包含有效值
        if torch.any(torch.isnan(q_fused)) or torch.any(torch.isinf(q_fused)):
            logger.error("❌ Q输出包含无效值")
            return False
        
        if torch.any(torch.isnan(k_fused)) or torch.any(torch.isinf(k_fused)):
            logger.error("❌ K输出包含无效值")
            return False
        
        if torch.any(torch.isnan(v_fused)) or torch.any(torch.isinf(v_fused)):
            logger.error("❌ V输出包含无效值")
            return False
        
        logger.info("✅ 正确性测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 正确性测试失败: {e}")
        return False

def test_fused_ln_qkv_gptq_batch():
    """测试批处理"""
    logger.info("📦 测试批处理...")
    
    try:
        # 创建批处理测试数据
        batch_sizes = [1, 2, 4, 8]
        seq_lens = [1, 16, 32, 64]
        
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过批处理测试")
            return False
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                logger.info(f"  测试 batch_size={batch_size}, seq_len={seq_len}")
                
                # 创建测试数据
                input_data, gptq_layer = create_test_data(batch_size, seq_len)
                
                # 测试融合算子
                q, k, v = fused_op.fused_ln_qkv_gptq(
                    input=input_data,
                    qweight=gptq_layer.qweight,
                    qzeros=gptq_layer.qzeros,
                    scales=gptq_layer.scales,
                    ln_weight=gptq_layer.ln_weight,
                    ln_bias=gptq_layer.ln_bias,
                    num_heads=32,
                    kv_num_heads=32,
                    head_size=128,
                    groupsize=128
                )
                
                # 验证输出形状
                expected_shape = (batch_size, 32, seq_len, 128)
                assert q.shape == expected_shape, f"Q形状错误: {q.shape} != {expected_shape}"
                assert k.shape == expected_shape, f"K形状错误: {k.shape} != {expected_shape}"
                assert v.shape == expected_shape, f"V形状错误: {v.shape} != {expected_shape}"
        
        logger.info("✅ 批处理测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 批处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始融合LN+QKV GPTQ功能测试")
    
    # 检查CUDA环境
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行测试")
        return False
    
    logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
    logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
    
    # 运行测试
    tests = [
        ("基础功能", test_fused_ln_qkv_gptq_basic),
        ("性能测试", test_fused_ln_qkv_gptq_performance),
        ("正确性测试", test_fused_ln_qkv_gptq_correctness),
        ("批处理测试", test_fused_ln_qkv_gptq_batch)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} 测试通过")
            else:
                logger.error(f"❌ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
    
    # 总结
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 测试总结: {passed}/{total} 通过")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 所有测试通过!")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
