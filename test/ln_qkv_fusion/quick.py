#!/usr/bin/env python3
"""
LN+QKV融合内核快速测试脚本
"""

import os
import sys
import torch
import time
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_compile():
    """测试编译"""
    try:
        logger.info("🔧 编译LN+QKV融合内核...")
        
        # 切换到cuda目录
        cuda_dir = Path(__file__).parent.parent.parent / "cuda"
        os.chdir(cuda_dir)
        
        from torch.utils.cpp_extension import load
        
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        logger.info("✅ 编译成功!")
        return kernel_module
        
    except Exception as e:
        logger.error(f"❌ 编译失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_functionality(kernel_module):
    """测试功能"""
    try:
        logger.info("🧪 进行功能测试...")
        
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
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (hidden_dim // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(hidden_dim // groupsize, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # 输出张量
        q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        
        # 调用内核
        kernel_module.fused_ln_qkv_gptq_cuda(
            input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
            q_output, k_output, v_output,
            batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
        )
        
        logger.info("✅ 功能测试成功!")
        logger.info(f"📊 Q输出形状: {q_output.shape}")
        logger.info(f"📊 K输出形状: {k_output.shape}")
        logger.info(f"📊 V输出形状: {v_output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance(kernel_module):
    """测试性能"""
    try:
        logger.info("⚡ 进行性能测试...")
        
        # 性能测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 1024
        num_heads, kv_num_heads, head_size = 16, 16, 64
        groupsize = 128
        eps = 1e-5
        num_iterations = 50
        
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
        for _ in range(5):
            kernel_module.fused_ln_qkv_gptq_cuda(
                input_tensor, qweight, qzeros, scales, ln_weight, ln_bias,
                q_output, k_output, v_output,
                batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
            )
        
        torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        for _ in range(num_iterations):
            kernel_module.fused_ln_qkv_gptq_cuda(
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

def main():
    """主函数"""
    logger.info("🚀 开始LN+QKV融合内核快速测试...")
    
    # 编译测试
    kernel_module = test_compile()
    if kernel_module is None:
        logger.error("❌ 编译失败，退出测试")
        return False
    
    # 功能测试
    if not test_functionality(kernel_module):
        logger.error("❌ 功能测试失败")
        return False
    
    # 性能测试
    if not test_performance(kernel_module):
        logger.error("❌ 性能测试失败")
        return False
    
    logger.info("🎉 所有测试通过!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
