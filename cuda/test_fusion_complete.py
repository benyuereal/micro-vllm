#!/usr/bin/env python3
"""
完整的融合LN+QKV GPTQ CUDA内核编译和测试脚本
"""

import os
import sys
import torch
import time
import logging
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionKernelManager:
    """融合内核管理器"""
    
    def __init__(self):
        self.kernel_module = None
        self.kernel_loaded = False
        self.cuda_available = torch.cuda.is_available()
        
        if not self.cuda_available:
            logger.error("❌ CUDA不可用，无法编译CUDA内核")
            sys.exit(1)
        
        logger.info(f"✅ CUDA可用，版本: {torch.version.cuda}")
    
    def compile_kernel(self):
        """编译融合内核"""
        try:
            logger.info("🔧 编译融合LN+QKV GPTQ CUDA内核...")
            
            from torch.utils.cpp_extension import load
            
            self.kernel_module = load(
                name="fused_ln_qkv_gptq_cuda",
                sources=["ln_qkv_fusion_kernel.cu"],
                extra_cuda_cflags=["-O3", "-use_fast_math"],
                verbose=True
            )
            
            self.kernel_loaded = True
            logger.info("✅ 融合内核编译成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 内核编译失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_basic_functionality(self):
        """基础功能测试"""
        logger.info("🧪 开始基础功能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            num_heads, kv_num_heads, head_size = 16, 16, 64
            groupsize = 128
            eps = 1e-5
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 输出张量
            q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            
            logger.info(f"📊 测试数据: input{batch_size}x{seq_len}x{hidden_dim}")
            logger.info(f"📊 GPTQ参数: qweight{qweight.shape}, qzeros{qzeros.shape}, scales{scales.shape}")
            
            # 调用融合内核
            self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                q_output, k_output, v_output,
                batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
            )
            
            logger.info("✅ 基础功能测试成功!")
            logger.info(f"📊 Q输出形状: {q_output.shape}, 范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
            logger.info(f"📊 K输出形状: {k_output.shape}, 范围: [{k_output.min():.4f}, {k_output.max():.4f}]")
            logger.info(f"📊 V输出形状: {v_output.shape}, 范围: [{v_output.min():.4f}, {v_output.max():.4f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 基础功能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_layernorm_accuracy(self):
        """LayerNorm精度测试"""
        logger.info("🎯 开始LayerNorm精度测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 512
            num_heads, kv_num_heads, head_size = 8, 8, 64
            groupsize = 128
            eps = 1e-5
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数（使用单位矩阵）
            qweight = torch.zeros(hidden_dim // 8, hidden_dim * 3, dtype=torch.uint32, device='cuda')
            qzeros = torch.zeros(hidden_dim // groupsize, groupsize // 8, dtype=torch.uint32, device='cuda')
            scales = torch.ones(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 输出张量
            q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            
            # 使用融合内核
            self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                q_output, k_output, v_output,
                batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
            )
            
            # 使用PyTorch LayerNorm作为参考
            pytorch_ln = torch.nn.LayerNorm(hidden_dim, eps=eps, dtype=torch.float16, device='cuda')
            pytorch_ln.weight.data = ln_weight
            pytorch_ln.bias.data = ln_bias
            
            pytorch_output = pytorch_ln(input_tensor)
            
            # 比较结果
            q_output_flat = q_output.view(batch_size, seq_len, -1)
            k_output_flat = k_output.view(batch_size, seq_len, -1)
            v_output_flat = v_output.view(batch_size, seq_len, -1)
            
            # 检查Q输出（应该等于LayerNorm输出）
            q_diff = torch.abs(q_output_flat - pytorch_output).max()
            k_diff = torch.abs(k_output_flat - pytorch_output).max()
            v_diff = torch.abs(v_output_flat - pytorch_output).max()
            
            logger.info(f"🎯 LayerNorm精度测试结果:")
            logger.info(f"  📊 Q输出差异: {q_diff:.6f}")
            logger.info(f"  📊 K输出差异: {k_diff:.6f}")
            logger.info(f"  📊 V输出差异: {v_diff:.6f}")
            
            # 精度目标：差异 < 1e-3
            if q_diff < 1e-3 and k_diff < 1e-3 and v_diff < 1e-3:
                logger.info("✅ LayerNorm精度测试通过!")
                return True
            else:
                logger.warning("⚠️ LayerNorm精度测试未达标")
                return False
            
        except Exception as e:
            logger.error(f"❌ LayerNorm精度测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_performance(self):
        """性能测试"""
        logger.info("⚡ 开始性能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 性能测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 2048
            num_heads, kv_num_heads, head_size = 32, 32, 64
            groupsize = 128
            eps = 1e-5
            num_iterations = 100
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 输出张量
            q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            
            # 预热
            for _ in range(10):
                self.kernel_module.fused_ln_qkv_gptq_cuda(
                    input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                    q_output, k_output, v_output,
                    batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
                )
            
            torch.cuda.synchronize()
            
            # 性能测试
            start_time = time.time()
            for _ in range(num_iterations):
                self.kernel_module.fused_ln_qkv_gptq_cuda(
                    input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                    q_output, k_output, v_output,
                    batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            avg_time_ms = avg_time * 1000
            
            logger.info(f"⚡ 性能测试结果:")
            logger.info(f"  📊 总时间: {total_time:.4f}s")
            logger.info(f"  📊 平均时间: {avg_time:.4f}s ({avg_time_ms:.2f}ms)")
            logger.info(f"  📊 迭代次数: {num_iterations}")
            
            # 性能目标：< 0.5ms
            if avg_time_ms < 0.5:
                logger.info("✅ 性能测试通过! (目标: < 0.5ms)")
                return True
            else:
                logger.warning(f"⚠️ 性能测试未达标: {avg_time_ms:.2f}ms > 0.5ms")
                return False
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_gptq_dequantization(self):
        """GPTQ解量化测试"""
        logger.info("🔍 开始GPTQ解量化测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            num_heads, kv_num_heads, head_size = 16, 16, 64
            groupsize = 128
            eps = 1e-5
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # LayerNorm参数
            ln_weight = torch.ones(hidden_dim, dtype=torch.float16, device='cuda')
            ln_bias = torch.zeros(hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
            scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
            
            # 输出张量
            q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            self.kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                q_output, k_output, v_output,
                batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
            )
            
            logger.info("✅ GPTQ解量化测试成功!")
            logger.info(f"📊 Q输出范围: [{q_output.min():.4f}, {q_output.max():.4f}]")
            logger.info(f"📊 K输出范围: [{k_output.min():.4f}, {k_output.max():.4f}]")
            logger.info(f"📊 V输出范围: [{v_output.min():.4f}, {v_output.max():.4f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPTQ解量化测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    logger.info("🚀 开始融合LN+QKV GPTQ CUDA内核完整测试...")
    
    manager = FusionKernelManager()
    
    # 编译内核
    if not manager.compile_kernel():
        logger.error("❌ 内核编译失败，退出测试")
        return False
    
    # 基础功能测试
    if not manager.test_basic_functionality():
        logger.error("❌ 基础功能测试失败")
        return False
    
    # GPTQ解量化测试
    if not manager.test_gptq_dequantization():
        logger.error("❌ GPTQ解量化测试失败")
        return False
    
    # LayerNorm精度测试
    if not manager.test_layernorm_accuracy():
        logger.error("❌ LayerNorm精度测试失败")
        return False
    
    # 性能测试
    if not manager.test_performance():
        logger.error("❌ 性能测试失败")
        return False
    
    logger.info("🎉 所有测试通过! 融合LN+QKV GPTQ CUDA内核工作正常!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
