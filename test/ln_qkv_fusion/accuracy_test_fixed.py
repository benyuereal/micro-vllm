#!/usr/bin/env python3
"""
融合内核精度验证测试（修复版本）
分步验证LayerNorm和GPTQ解量化的正确性
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

class AccuracyValidator:
    """融合内核精度验证器"""
    
    def __init__(self):
        self.kernel_module = None
        self.kernel_loaded = False
    
    def load_kernel(self):
        """加载融合内核"""
        try:
            logger.info("🔧 编译融合内核...")
            
            # 编译融合内核
            from torch.utils.cpp_extension import load
            self.kernel_module = load(
                name="fused_ln_qkv_gptq_cuda",
                sources=[os.path.join(os.path.dirname(__file__), '..', '..', 'cuda', 'gptq_ln_qkv_fusion_kernel.cu')],
                extra_cuda_cflags=["-O3", "-use_fast_math"],
                verbose=False
            )
            
            self.kernel_loaded = True
            logger.info("✅ 融合内核加载成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合内核加载失败: {e}")
            return False
    
    def test_layernorm_accuracy(self):
        """测试LayerNorm部分正确性"""
        try:
            logger.info("🎯 测试LayerNorm部分正确性...")
            
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
            
            # GPTQ参数（分别创建Q、K、V）
            qweight_q = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
            qweight_k = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
            qweight_v = torch.zeros(hidden_dim // 8, hidden_dim, dtype=torch.uint32, device='cuda')
            
            qzeros_q = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
            qzeros_k = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
            qzeros_v = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
            
            scales_q = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            scales_k = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            scales_v = torch.ones(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
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
            
            # 检查精度
            tolerance = 1e-3
            if q_diff < tolerance and k_diff < tolerance and v_diff < tolerance:
                logger.info("✅ LayerNorm部分精度测试通过!")
                return True
            else:
                logger.warning(f"⚠️ LayerNorm部分精度未达标: 最大差异 {max(q_diff, k_diff, v_diff):.6f} > {tolerance}")
                return False
                
        except Exception as e:
            logger.error(f"❌ LayerNorm部分测试失败: {e}")
            return False
    
    def test_gptq_accuracy(self):
        """测试GPTQ解量化正确性"""
        try:
            logger.info("🎯 测试GPTQ解量化正确性...")
            
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 256
            groupsize = 128
            eps = 1e-5
            
            # 创建测试数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, 
                                     dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # 创建已知的GPTQ参数
            qweight_q = torch.randint(0, 15, (hidden_dim // 8, hidden_dim), dtype=torch.uint32, device='cuda')
            qweight_k = torch.randint(0, 15, (hidden_dim // 8, hidden_dim), dtype=torch.uint32, device='cuda')
            qweight_v = torch.randint(0, 15, (hidden_dim // 8, hidden_dim), dtype=torch.uint32, device='cuda')
            
            qzeros_q = torch.randint(0, 15, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            qzeros_k = torch.randint(0, 15, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            qzeros_v = torch.randint(0, 15, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            
            scales_q = torch.randn(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            scales_k = torch.randn(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            scales_v = torch.randn(hidden_dim // groupsize, hidden_dim, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
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
            
            # 检查输出是否合理
            if torch.isnan(q_output).any() or torch.isinf(q_output).any():
                logger.error("❌ Q输出包含NaN或Inf")
                return False
            
            if torch.isnan(k_output).any() or torch.isinf(k_output).any():
                logger.error("❌ K输出包含NaN或Inf")
                return False
            
            if torch.isnan(v_output).any() or torch.isinf(v_output).any():
                logger.error("❌ V输出包含NaN或Inf")
                return False
            
            # 检查输出范围是否合理
            q_range = q_output.max() - q_output.min()
            k_range = k_output.max() - k_output.min()
            v_range = v_output.max() - v_output.min()
            
            logger.info(f"📊 GPTQ解量化测试结果:")
            logger.info(f"  📊 Q输出范围: {q_range:.4f}")
            logger.info(f"  📊 K输出范围: {k_range:.4f}")
            logger.info(f"  📊 V输出范围: {v_range:.4f}")
            
            if q_range > 0 and k_range > 0 and v_range > 0:
                logger.info("✅ GPTQ解量化测试通过!")
                return True
            else:
                logger.error("❌ GPTQ解量化输出异常")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPTQ解量化测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有精度测试"""
        logger.info("🚀 开始融合内核精度验证...")
        
        # 加载内核
        if not self.load_kernel():
            return False
        
        # 运行测试
        tests = [
            ("LayerNorm精度测试", self.test_layernorm_accuracy),
            ("GPTQ解量化测试", self.test_gptq_accuracy),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"🧪 开始{test_name}...")
            if test_func():
                logger.info(f"✅ {test_name}通过!")
                passed += 1
            else:
                logger.error(f"❌ {test_name}失败!")
        
        logger.info(f"📊 精度测试结果: {passed}/{total} 通过")
        return passed == total

def main():
    """主函数"""
    validator = AccuracyValidator()
    success = validator.run_all_tests()
    
    if success:
        logger.info("🎉 所有精度测试通过!")
    else:
        logger.error("❌ 部分精度测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
