#!/usr/bin/env python3
"""
融合内核功能测试（修复版本）
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

class FusionTester:
    """融合内核测试器"""
    
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
    
    def test_fusion_functionality(self):
        """测试融合功能"""
        try:
            logger.info("🧪 开始融合功能测试...")
            
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            groupsize = 128
            eps = 1e-5
            
            logger.info(f"📊 测试数据: input{batch_size}x{seq_len}x{hidden_dim}")
            
            # 创建随机输入
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, 
                                     dtype=torch.float16, device='cuda')
            
            # 创建GPTQ参数（分别创建Q、K、V）
            qweight_q = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            qweight_k = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            qweight_v = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            
            qzeros_q = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            qzeros_k = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            qzeros_v = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            
            scales_q = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            scales_k = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            scales_v = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            
            # 创建LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            logger.info(f"📊 GPTQ参数: qweight_q{qweight_q.shape}, qweight_k{qweight_k.shape}, qweight_v{qweight_v.shape}")
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight_q, qweight_k, qweight_v,
                qzeros_q, qzeros_k, qzeros_v,
                scales_q, scales_k, scales_v,
                ln_weight, ln_bias,
                batch_size, seq_len, hidden_dim, groupsize, eps
            )
            
            # 解包QKV
            q_output = qkv_output[0]
            k_output = qkv_output[1]
            v_output = qkv_output[2]
            
            logger.info(f"📊 Q输出形状: {q_output.shape}, 范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
            logger.info(f"📊 K输出形状: {k_output.shape}, 范围: [{k_output.min():.4f}, {k_output.max():.4f}]")
            logger.info(f"📊 V输出形状: {v_output.shape}, 范围: [{v_output.min():.4f}, {v_output.max():.4f}]")
            
            # 检查输出形状
            expected_shape = (batch_size, seq_len, hidden_dim)
            if q_output.shape != expected_shape or k_output.shape != expected_shape or v_output.shape != expected_shape:
                logger.error(f"❌ 输出形状错误: 期望 {expected_shape}, 实际 Q{q_output.shape}, K{k_output.shape}, V{v_output.shape}")
                return False
            
            # 检查输出是否全零
            if torch.allclose(q_output, torch.zeros_like(q_output)) and \
               torch.allclose(k_output, torch.zeros_like(k_output)) and \
               torch.allclose(v_output, torch.zeros_like(v_output)):
                logger.error("❌ 输出全零")
                return False
            
            # 检查NaN和Inf
            if torch.isnan(q_output).any() or torch.isinf(q_output).any() or \
               torch.isnan(k_output).any() or torch.isinf(k_output).any() or \
               torch.isnan(v_output).any() or torch.isinf(v_output).any():
                logger.error("❌ 输出包含NaN或Inf")
                return False
            
            logger.info("✅ 融合功能测试成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合功能测试失败: {e}")
            return False
    
    def test_fusion_output_validation(self):
        """测试融合输出验证"""
        try:
            logger.info("🎯 开始融合内核输出验证...")
            
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 512
            groupsize = 128
            eps = 1e-5
            
            # 创建随机输入
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, 
                                     dtype=torch.float16, device='cuda')
            
            # 创建GPTQ参数
            qweight_q = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            qweight_k = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            qweight_v = torch.randint(0, 15, (hidden_dim//8, hidden_dim), 
                                    dtype=torch.uint32, device='cuda')
            
            qzeros_q = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            qzeros_k = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            qzeros_v = torch.randint(0, 15, (8, 16), 
                                   dtype=torch.uint32, device='cuda')
            
            scales_q = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            scales_k = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            scales_v = torch.randn(8, hidden_dim, 
                                 dtype=torch.float16, device='cuda')
            
            # 创建LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight_q, qweight_k, qweight_v,
                qzeros_q, qzeros_k, qzeros_v,
                scales_q, scales_k, scales_v,
                ln_weight, ln_bias,
                batch_size, seq_len, hidden_dim, groupsize, eps
            )
            
            # 解包QKV
            q_output = qkv_output[0]
            k_output = qkv_output[1]
            v_output = qkv_output[2]
            
            logger.info("🎯 融合内核输出验证:")
            logger.info(f"  📊 Q输出形状: {q_output.shape}, 范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
            logger.info(f"  📊 K输出形状: {k_output.shape}, 范围: [{k_output.min():.4f}, {k_output.max():.4f}]")
            logger.info(f"  📊 V输出形状: {v_output.shape}, 范围: [{v_output.min():.4f}, {v_output.max():.4f}]")
            
            # 检查输出是否全零
            if torch.allclose(q_output, torch.zeros_like(q_output)) and \
               torch.allclose(k_output, torch.zeros_like(k_output)) and \
               torch.allclose(v_output, torch.zeros_like(v_output)):
                logger.error("❌ 输出全零")
                return False
            
            logger.info("✅ 融合内核输出验证成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合内核输出验证失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始融合内核测试...")
        
        # 加载内核
        if not self.load_kernel():
            return False
        
        # 运行测试
        tests = [
            ("融合功能测试", self.test_fusion_functionality),
            ("融合输出验证", self.test_fusion_output_validation),
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
        
        logger.info(f"📊 测试结果: {passed}/{total} 通过")
        return passed == total

def main():
    """主函数"""
    tester = FusionTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("🎉 所有测试通过!")
    else:
        logger.error("❌ 部分测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
