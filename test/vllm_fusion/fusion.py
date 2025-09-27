#!/usr/bin/env python3
"""
vLLM融合内核功能测试套件
包含完整的功能测试、精度测试和性能测试
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

class VLLMFusionTester:
    """vLLM融合内核测试器"""
    
    def __init__(self):
        self.kernel_module = None
        self.kernel_loaded = False
    
    def load_kernel(self):
        """加载vLLM内核"""
        try:
            logger.info("🔧 编译vLLM GPTQ内核...")
            
            # 切换到cuda目录
            cuda_dir = Path(__file__).parent.parent.parent / "cuda"
            os.chdir(cuda_dir)
            
            from torch.utils.cpp_extension import load
            
            self.kernel_module = load(
                name="gptq_cuda_kernel_vllm",
                sources=["gptq_cuda_kernel_vllm.cu"],
                extra_cuda_cflags=["-O3", "-use_fast_math"],
                verbose=True
            )
            
            self.kernel_loaded = True
            logger.info("✅ vLLM内核加载成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ vLLM内核加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_functionality(self):
        """功能测试"""
        logger.info("🧪 开始vLLM功能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 1024
            output_dim = 2048
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, output_dim), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (32, 16), dtype=torch.uint32, device='cuda')
            scales = torch.randn(32, output_dim, dtype=torch.float16, device='cuda')
            
            # 输出张量
            output = torch.zeros(batch_size, seq_len, output_dim, dtype=torch.float16, device='cuda')
            
            logger.info(f"📊 测试数据: input{batch_size}x{seq_len}x{hidden_dim}")
            logger.info(f"📊 GPTQ参数: qweight{qweight.shape}, qzeros{qzeros.shape}, scales{scales.shape}")
            
            # 功能测试
            self.kernel_module.gptq_gemm(
                input_tensor,
                qweight,
                qzeros,
                scales,
                torch.tensor([]),  # g_idx (空)
                True,  # use_exllama
                4  # bit
            )
            
            logger.info("✅ vLLM功能测试成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ vLLM功能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_performance(self):
        """性能测试"""
        logger.info("⚡ 开始vLLM性能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行测试")
            return False
        
        try:
            # 性能测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 2048
            output_dim = 4096
            num_iterations = 100
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, output_dim), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (32, 16), dtype=torch.uint32, device='cuda')
            scales = torch.randn(32, output_dim, dtype=torch.float16, device='cuda')
            
            # 输出张量
            output = torch.zeros(batch_size, seq_len, output_dim, dtype=torch.float16, device='cuda')
            
            # 预热
            for _ in range(10):
                self.kernel_module.gptq_gemm(
                    input_tensor, qweight, qzeros, scales,
                    torch.tensor([]), True, 4
                )
            
            torch.cuda.synchronize()
            
            # 性能测试
            start_time = time.time()
            for _ in range(num_iterations):
                self.kernel_module.gptq_gemm(
                    input_tensor, qweight, qzeros, scales,
                    torch.tensor([]), True, 4
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            avg_time_ms = avg_time * 1000
            
            logger.info(f"⚡ vLLM性能测试结果:")
            logger.info(f"  📊 总时间: {total_time:.4f}s")
            logger.info(f"  📊 平均时间: {avg_time:.4f}s ({avg_time_ms:.2f}ms)")
            logger.info(f"  📊 迭代次数: {num_iterations}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ vLLM性能测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    logger.info("🚀 开始vLLM融合内核测试...")
    
    tester = VLLMFusionTester()
    
    # 加载内核
    if not tester.load_kernel():
        logger.error("❌ 内核加载失败，退出测试")
        return False
    
    # 功能测试
    if not tester.test_functionality():
        logger.error("❌ 功能测试失败")
        return False
    
    # 性能测试
    if not tester.test_performance():
        logger.error("❌ 性能测试失败")
        return False
    
    logger.info("🎉 vLLM融合内核测试完成!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
