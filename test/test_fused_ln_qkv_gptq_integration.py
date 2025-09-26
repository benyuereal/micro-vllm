#!/usr/bin/env python3
"""
融合LN+QKV GPTQ集成测试
测试融合算子在实际Layer中的集成效果
"""

import sys
import os
import torch
import time
import logging
import numpy as np
from typing import Tuple, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ, fused_ln_qkv_gptq
from core.layer.optimized_qwen_layer import OptimizedQwenModelLayerAdapter

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

class MockLayer:
    """模拟Layer"""
    
    def __init__(self, hidden_dim: int = 4096):
        self.hidden_dim = hidden_dim
        
        # 创建模拟的LayerNorm
        self.ln_1 = torch.nn.LayerNorm(hidden_dim)
        self.ln_1.weight.data = torch.randn(hidden_dim, dtype=torch.float16)
        self.ln_1.bias.data = torch.randn(hidden_dim, dtype=torch.float16)
        
        # 创建模拟的QKV投影
        self.attn = MockGPTQLayer(hidden_dim, hidden_dim * 3, groupsize=128)
        
        # 创建模拟的输出投影
        self.c_proj = MockGPTQLayer(hidden_dim, hidden_dim, groupsize=128)
        
        # 创建模拟的MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim)
        )

def create_test_data(batch_size: int = 1, seq_len: int = 1, hidden_dim: int = 4096) -> Tuple[torch.Tensor, MockLayer]:
    """创建测试数据"""
    # 创建输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
    
    # 创建模拟的Layer
    layer = MockLayer(hidden_dim)
    
    return input_data, layer

def test_fused_ln_qkv_integration():
    """测试融合LN+QKV集成"""
    logger.info("🧪 测试融合LN+QKV集成...")
    
    try:
        # 创建测试数据
        input_data, layer = create_test_data()
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过集成测试")
            return False
        
        # 使用融合算子
        q, k, v = fused_op.fused_ln_qkv_gptq(
            input=input_data,
            qweight=layer.attn.qweight,
            qzeros=layer.attn.qzeros,
            scales=layer.attn.scales,
            ln_weight=layer.ln_1.weight,
            ln_bias=layer.ln_1.bias,
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
        
        logger.info("✅ 融合LN+QKV集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 融合LN+QKV集成测试失败: {e}")
        return False

def test_layer_performance_comparison():
    """测试Layer性能对比"""
    logger.info("🚀 测试Layer性能对比...")
    
    try:
        # 创建测试数据
        input_data, layer = create_test_data()
        
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过性能对比测试")
            return False
        
        # 测试融合算子性能
        logger.info("🔥 测试融合算子性能...")
        fused_times = []
        for _ in range(100):
            start_time = time.time()
            q, k, v = fused_op.fused_ln_qkv_gptq(
                input=input_data,
                qweight=layer.attn.qweight,
                qzeros=layer.attn.qzeros,
                scales=layer.attn.scales,
                ln_weight=layer.ln_1.weight,
                ln_bias=layer.ln_1.bias,
                num_heads=32,
                kv_num_heads=32,
                head_size=128,
                groupsize=128
            )
            end_time = time.time()
            fused_times.append((end_time - start_time) * 1000)
        
        # 测试分离算子性能
        logger.info("🔧 测试分离算子性能...")
        separate_times = []
        for _ in range(100):
            start_time = time.time()
            
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_data, 
                input_data.shape[-1:], 
                layer.ln_1.weight, 
                layer.ln_1.bias
            )
            
            # QKV投影（使用随机权重）
            qkv_weight = torch.randn(input_data.shape[-1], input_data.shape[-1] * 3, dtype=torch.float16)
            qkv_output = torch.matmul(ln_output, qkv_weight)
            
            # 分割QKV
            hidden_size = qkv_output.shape[-1] // 3
            q, k, v = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            batch_size, seq_len, hidden_dim = q.shape
            q = q.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            
            end_time = time.time()
            separate_times.append((end_time - start_time) * 1000)
        
        # 计算性能统计
        fused_avg = np.mean(fused_times)
        separate_avg = np.mean(separate_times)
        speedup = separate_avg / fused_avg
        
        logger.info(f"📊 性能对比结果:")
        logger.info(f"  融合算子: {fused_avg:.2f}ms")
        logger.info(f"  分离算子: {separate_avg:.2f}ms")
        logger.info(f"  加速比: {speedup:.2f}x")
        
        # 性能目标检查
        target_time = 0.25  # 目标时间: 0.25ms
        if fused_avg <= target_time:
            logger.info(f"🎯 达到性能目标: {fused_avg:.2f}ms <= {target_time}ms")
            return True
        else:
            logger.warning(f"⚠️ 未达到性能目标: {fused_avg:.2f}ms > {target_time}ms")
            return False
        
    except Exception as e:
        logger.error(f"❌ Layer性能对比测试失败: {e}")
        return False

def test_batch_processing():
    """测试批处理性能"""
    logger.info("📦 测试批处理性能...")
    
    try:
        # 测试不同的批处理配置
        batch_configs = [
            {"batch_size": 1, "seq_len": 1, "name": "单token"},
            {"batch_size": 1, "seq_len": 16, "name": "短序列"},
            {"batch_size": 1, "seq_len": 64, "name": "中序列"},
            {"batch_size": 4, "seq_len": 16, "name": "批处理"},
        ]
        
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用，跳过批处理测试")
            return False
        
        for config in batch_configs:
            logger.info(f"  测试配置: {config['name']}")
            
            # 创建测试数据
            input_data, layer = create_test_data(
                config['batch_size'], 
                config['seq_len']
            )
            
            # 测试性能
            times = []
            for _ in range(50):
                start_time = time.time()
                q, k, v = fused_op.fused_ln_qkv_gptq(
                    input=input_data,
                    qweight=layer.attn.qweight,
                    qzeros=layer.attn.qzeros,
                    scales=layer.attn.scales,
                    ln_weight=layer.ln_1.weight,
                    ln_bias=layer.ln_1.bias,
                    num_heads=32,
                    kv_num_heads=32,
                    head_size=128,
                    groupsize=128
                )
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = np.mean(times)
            logger.info(f"    平均时间: {avg_time:.2f}ms")
            
            # 验证输出形状
            expected_shape = (config['batch_size'], 32, config['seq_len'], 128)
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
    logger.info("🚀 开始融合LN+QKV GPTQ集成测试")
    
    # 检查CUDA环境
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行测试")
        return False
    
    logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
    logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
    
    # 运行测试
    tests = [
        ("融合LN+QKV集成", test_fused_ln_qkv_integration),
        ("Layer性能对比", test_layer_performance_comparison),
        ("批处理测试", test_batch_processing)
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
        logger.info("🎉 所有集成测试通过!")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
