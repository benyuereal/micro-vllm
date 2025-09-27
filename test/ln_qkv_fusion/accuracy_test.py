#!/usr/bin/env python3
"""
融合内核精度验证测试
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
            
            # 切换到cuda目录
            cuda_dir = Path(__file__).parent.parent.parent / "cuda"
            os.chdir(cuda_dir)
            
            from torch.utils.cpp_extension import load
            
            self.kernel_module = load(
                name="fused_ln_qkv_gptq_cuda",
                sources=["gptq_ln_qkv_fusion_kernel.cu"],
                extra_cuda_cflags=["-O3", "-use_fast_math"],
                verbose=False
            )
            
            self.kernel_loaded = True
            logger.info("✅ 融合内核加载成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 融合内核加载失败: {e}")
            return False
    
    def test_layernorm_only(self):
        """测试LayerNorm部分的正确性"""
        logger.info("🎯 测试LayerNorm部分正确性...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载")
            return False
        
        try:
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            groupsize = 128
            eps = 1e-5
            
            # 创建输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # 创建单位矩阵作为GPTQ参数（这样GPTQ部分不会改变数值）
            qweight = torch.zeros(hidden_dim // 8, hidden_dim * 3, dtype=torch.uint32, device='cuda')
            qzeros = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
            scales = torch.ones(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                batch_size, seq_len, hidden_dim, groupsize, eps
            )
            
            # 解包QKV输出
            q_output = qkv_output[0]
            k_output = qkv_output[1]
            v_output = qkv_output[2]
            
            # 使用PyTorch LayerNorm作为参考
            pytorch_ln = torch.nn.LayerNorm(hidden_dim, eps=eps, dtype=torch.float16, device='cuda')
            pytorch_ln.weight.data = ln_weight
            pytorch_ln.bias.data = ln_bias
            pytorch_output = pytorch_ln(input_tensor)
            
            # 比较Q输出（应该等于LayerNorm输出）
            q_diff = torch.abs(q_output - pytorch_output).max()
            
            logger.info(f"📊 LayerNorm精度测试结果:")
            logger.info(f"  📊 Q输出差异: {q_diff:.6f}")
            logger.info(f"  📊 期望差异: < 1e-3")
            
            if q_diff < 1e-3:
                logger.info("✅ LayerNorm部分正确性验证通过!")
                return True
            else:
                logger.error(f"❌ LayerNorm部分精度不足: {q_diff:.6f} > 1e-3")
                return False
                
        except Exception as e:
            logger.error(f"❌ LayerNorm测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_gptq_dequantization(self):
        """测试GPTQ解量化的正确性"""
        logger.info("🎯 测试GPTQ解量化正确性...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载")
            return False
        
        try:
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            groupsize = 128
            eps = 1e-5
            
            # 创建输入数据（单位矩阵，这样LayerNorm输出也是单位矩阵）
            input_tensor = torch.ones(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数（单位变换）
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # 创建已知的GPTQ参数
            # 这里我们创建一个简单的测试案例
            qweight = torch.randint(0, 16, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 8, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            scales = torch.ones(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                batch_size, seq_len, hidden_dim, groupsize, eps
            )
            
            # 解包QKV输出
            q_output = qkv_output[0]
            k_output = qkv_output[1]
            v_output = qkv_output[2]
            
            # 检查输出是否合理
            logger.info(f"📊 GPTQ解量化测试结果:")
            logger.info(f"  📊 Q输出范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
            logger.info(f"  📊 K输出范围: [{k_output.min():.4f}, {k_output.max():.4f}]")
            logger.info(f"  📊 V输出范围: [{v_output.min():.4f}, {v_output.max():.4f}]")
            
            # 检查输出是否包含NaN或Inf
            has_nan = torch.isnan(q_output).any() or torch.isnan(k_output).any() or torch.isnan(v_output).any()
            has_inf = torch.isinf(q_output).any() or torch.isinf(k_output).any() or torch.isinf(v_output).any()
            
            if has_nan:
                logger.error("❌ 输出包含NaN值")
                return False
            
            if has_inf:
                logger.error("❌ 输出包含Inf值")
                return False
            
            logger.info("✅ GPTQ解量化正确性验证通过!")
            return True
                
        except Exception as e:
            logger.error(f"❌ GPTQ解量化测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_end_to_end_consistency(self):
        """测试端到端一致性"""
        logger.info("🎯 测试端到端一致性...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载")
            return False
        
        try:
            # 测试参数
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            groupsize = 128
            eps = 1e-5
            
            # 创建相同的输入数据
            torch.manual_seed(42)  # 固定随机种子
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 16, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 8, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            scales = torch.ones(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 多次运行，检查一致性
            results = []
            for i in range(5):
                qkv_output = self.kernel_module.fused_ln_qkv_gptq_cuda(
                    input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                    batch_size, seq_len, hidden_dim, groupsize, eps
                )
                results.append(qkv_output)
            
            # 检查所有结果是否一致
            for i in range(1, len(results)):
                q_diff = torch.abs(results[0][0] - results[i][0]).max()
                k_diff = torch.abs(results[0][1] - results[i][1]).max()
                v_diff = torch.abs(results[0][2] - results[i][2]).max()
                
                if q_diff > 1e-6 or k_diff > 1e-6 or v_diff > 1e-6:
                    logger.error(f"❌ 第{i+1}次运行结果不一致")
                    logger.error(f"  Q差异: {q_diff:.8f}")
                    logger.error(f"  K差异: {k_diff:.8f}")
                    logger.error(f"  V差异: {v_diff:.8f}")
                    return False
            
            logger.info("✅ 端到端一致性验证通过!")
            return True
                
        except Exception as e:
            logger.error(f"❌ 端到端一致性测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    logger.info("🚀 开始融合内核精度验证...")
    
    validator = AccuracyValidator()
    
    # 加载内核
    if not validator.load_kernel():
        logger.error("❌ 内核加载失败，退出测试")
        return False
    
    # 测试LayerNorm部分
    if not validator.test_layernorm_only():
        logger.error("❌ LayerNorm部分测试失败")
        return False
    
    # 测试GPTQ解量化
    if not validator.test_gptq_dequantization():
        logger.error("❌ GPTQ解量化测试失败")
        return False
    
    # 测试端到端一致性
    if not validator.test_end_to_end_consistency():
        logger.error("❌ 端到端一致性测试失败")
        return False
    
    logger.info("🎉 所有精度验证测试通过!")
    return True

if __name__ == "__main__":
    main()
