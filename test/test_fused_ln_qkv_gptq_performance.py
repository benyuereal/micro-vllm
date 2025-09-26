#!/usr/bin/env python3
"""
融合LN+QKV GPTQ性能测试
对比融合算子与分离算子的性能
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
from core.layer.gptq import GPTQCUDAFusion

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

def benchmark_fused_ln_qkv_gptq(input_data: torch.Tensor, gptq_layer: MockGPTQLayer, num_runs: int = 100) -> float:
    """测试融合算子性能"""
    try:
        # 创建融合算子
        fused_op = FusedLNQKVGPTQ()
        
        if not fused_op.is_available():
            logger.warning("⚠️ 融合算子不可用")
            return float('inf')
        
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
        
        return np.mean(times)
        
    except Exception as e:
        logger.error(f"❌ 融合算子性能测试失败: {e}")
        return float('inf')

def benchmark_separate_ln_qkv_gptq(input_data: torch.Tensor, gptq_layer: MockGPTQLayer, num_runs: int = 100) -> float:
    """测试分离算子性能"""
    try:
        # 创建GPTQ融合算子
        gptq_fusion = GPTQCUDAFusion()
        
        # 预热
        for _ in range(10):
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_data, 
                input_data.shape[-1:], 
                gptq_layer.ln_weight, 
                gptq_layer.ln_bias
            )
            
            # QKV投影
            batch_size, seq_len, hidden_dim = ln_output.shape
            input_2d = ln_output.view(-1, hidden_dim)
            
            qkv_output = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_2d,
                qweight=gptq_layer.qweight,
                qzeros=gptq_layer.qzeros,
                scales=gptq_layer.scales
            )
            
            # 重塑和分割
            qkv_output = qkv_output.view(batch_size, seq_len, -1)
            hidden_size = qkv_output.shape[-1] // 3
            q, k, v = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            q = q.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
        
        # 性能测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_data, 
                input_data.shape[-1:], 
                gptq_layer.ln_weight, 
                gptq_layer.ln_bias
            )
            
            # QKV投影
            batch_size, seq_len, hidden_dim = ln_output.shape
            input_2d = ln_output.view(-1, hidden_dim)
            
            qkv_output = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_2d,
                qweight=gptq_layer.qweight,
                qzeros=gptq_layer.qzeros,
                scales=gptq_layer.scales
            )
            
            # 重塑和分割
            qkv_output = qkv_output.view(batch_size, seq_len, -1)
            hidden_size = qkv_output.shape[-1] // 3
            q, k, v = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            q = q.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        return np.mean(times)
        
    except Exception as e:
        logger.error(f"❌ 分离算子性能测试失败: {e}")
        return float('inf')

def benchmark_pytorch_ln_qkv_gptq(input_data: torch.Tensor, gptq_layer: MockGPTQLayer, num_runs: int = 100) -> float:
    """测试PyTorch实现性能"""
    try:
        # 预热
        for _ in range(10):
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
            q, k, v = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            batch_size, seq_len, hidden_dim = q.shape
            q = q.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
        
        # 性能测试
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_data, 
                input_data.shape[-1:], 
                gptq_layer.ln_weight, 
                gptq_layer.ln_bias
            )
            
            # 简化的QKV投影
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
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        return np.mean(times)
        
    except Exception as e:
        logger.error(f"❌ PyTorch实现性能测试失败: {e}")
        return float('inf')

def run_performance_comparison():
    """运行性能对比测试"""
    logger.info("🚀 开始性能对比测试")
    
    # 测试配置
    test_configs = [
        {"batch_size": 1, "seq_len": 1, "hidden_dim": 4096, "name": "单token"},
        {"batch_size": 1, "seq_len": 16, "hidden_dim": 4096, "name": "短序列"},
        {"batch_size": 1, "seq_len": 64, "hidden_dim": 4096, "name": "中序列"},
        {"batch_size": 1, "seq_len": 256, "hidden_dim": 4096, "name": "长序列"},
        {"batch_size": 4, "seq_len": 16, "hidden_dim": 4096, "name": "批处理"},
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 测试配置: {config['name']}")
        logger.info(f"   batch_size: {config['batch_size']}")
        logger.info(f"   seq_len: {config['seq_len']}")
        logger.info(f"   hidden_dim: {config['hidden_dim']}")
        logger.info(f"{'='*60}")
        
        # 创建测试数据
        input_data, gptq_layer = create_test_data(
            config['batch_size'], 
            config['seq_len'], 
            config['hidden_dim']
        )
        
        # 测试融合算子
        logger.info("🔥 测试融合算子...")
        fused_time = benchmark_fused_ln_qkv_gptq(input_data, gptq_layer)
        
        # 测试分离算子
        logger.info("🔧 测试分离算子...")
        separate_time = benchmark_separate_ln_qkv_gptq(input_data, gptq_layer)
        
        # 测试PyTorch实现
        logger.info("🐍 测试PyTorch实现...")
        pytorch_time = benchmark_pytorch_ln_qkv_gptq(input_data, gptq_layer)
        
        # 计算加速比
        fused_speedup = separate_time / fused_time if fused_time != float('inf') else 0
        pytorch_speedup = pytorch_time / fused_time if fused_time != float('inf') else 0
        
        # 记录结果
        result = {
            "config": config['name'],
            "fused_time": fused_time,
            "separate_time": separate_time,
            "pytorch_time": pytorch_time,
            "fused_speedup": fused_speedup,
            "pytorch_speedup": pytorch_speedup
        }
        results.append(result)
        
        # 打印结果
        logger.info(f"📊 性能结果:")
        logger.info(f"  融合算子: {fused_time:.2f}ms")
        logger.info(f"  分离算子: {separate_time:.2f}ms")
        logger.info(f"  PyTorch: {pytorch_time:.2f}ms")
        logger.info(f"  融合加速: {fused_speedup:.2f}x")
        logger.info(f"  PyTorch加速: {pytorch_speedup:.2f}x")
        
        # 性能目标检查
        target_time = 0.25  # 目标时间: 0.25ms
        if fused_time <= target_time:
            logger.info(f"🎯 达到性能目标: {fused_time:.2f}ms <= {target_time}ms")
        else:
            logger.warning(f"⚠️ 未达到性能目标: {fused_time:.2f}ms > {target_time}ms")
    
    return results

def print_summary(results: List[dict]):
    """打印测试总结"""
    logger.info(f"\n{'='*80}")
    logger.info("📊 性能测试总结")
    logger.info(f"{'='*80}")
    
    # 计算平均加速比
    valid_results = [r for r in results if r['fused_time'] != float('inf')]
    
    if valid_results:
        avg_fused_speedup = np.mean([r['fused_speedup'] for r in valid_results])
        avg_pytorch_speedup = np.mean([r['pytorch_speedup'] for r in valid_results])
        
        logger.info(f"📈 平均加速比:")
        logger.info(f"  融合 vs 分离: {avg_fused_speedup:.2f}x")
        logger.info(f"  融合 vs PyTorch: {avg_pytorch_speedup:.2f}x")
        
        # 最佳性能
        best_result = min(valid_results, key=lambda x: x['fused_time'])
        logger.info(f"🏆 最佳性能: {best_result['config']} - {best_result['fused_time']:.2f}ms")
        
        # 性能目标达成情况
        target_achieved = sum(1 for r in valid_results if r['fused_time'] <= 0.25)
        total_tests = len(valid_results)
        logger.info(f"🎯 性能目标达成: {target_achieved}/{total_tests}")
        
        if target_achieved == total_tests:
            logger.info("🎉 所有测试都达到性能目标!")
        else:
            logger.warning(f"⚠️ {total_tests - target_achieved} 个测试未达到性能目标")
    
    else:
        logger.error("❌ 没有有效的测试结果")

def main():
    """主测试函数"""
    logger.info("🚀 开始融合LN+QKV GPTQ性能测试")
    
    # 检查CUDA环境
    if not torch.cuda.is_available():
        logger.error("❌ CUDA不可用，无法运行测试")
        return False
    
    logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
    logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
    
    # 运行性能对比测试
    results = run_performance_comparison()
    
    # 打印总结
    print_summary(results)
    
    # 检查是否有有效的测试结果
    valid_results = [r for r in results if r['fused_time'] != float('inf')]
    
    if valid_results:
        logger.info("🎉 性能测试完成!")
        return True
    else:
        logger.error("❌ 没有有效的测试结果")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
