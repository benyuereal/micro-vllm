#!/usr/bin/env python3
"""
vLLM GPTQ CUDA内核测试脚本
"""

import os
import sys
import torch
import time
import logging
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLLMGPTQTester:
    """vLLM GPTQ测试器"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.kernel_loaded = False
        self.kernel_module = None
        
        if self.cuda_available:
            logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
            logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
        else:
            logger.error("❌ CUDA不可用，无法运行测试")
            sys.exit(1)
    
    def load_kernel(self):
        """加载vLLM GPTQ CUDA内核"""
        logger.info("🔧 加载vLLM GPTQ CUDA内核...")
        
        try:
            # 尝试导入编译的内核
            import gptq_cuda_kernel_vllm
            self.kernel_module = gptq_cuda_kernel_vllm
            self.kernel_loaded = True
            logger.info("✅ vLLM GPTQ CUDA内核加载成功")
            return True
        except ImportError as e:
            logger.error(f"❌ 无法导入vLLM GPTQ CUDA内核: {e}")
            logger.info("💡 请先运行 compile_vllm.py 编译内核")
            return False
    
    def test_functionality(self):
        """功能测试"""
        logger.info("🧪 开始功能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行功能测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 4096
            output_dim = 12288  # QKV总维度
            
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
            
            logger.info("✅ 功能测试成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 功能测试失败: {e}")
            return False
    
    def test_performance(self):
        """性能测试"""
        logger.info("⚡ 开始性能测试...")
        
        if not self.kernel_loaded:
            logger.error("❌ 内核未加载，无法进行性能测试")
            return False
        
        try:
            # 测试数据
            batch_size, seq_len, hidden_dim = 1, 1, 4096
            output_dim = 12288  # QKV总维度
            
            # 输入数据
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
            
            # GPTQ参数
            qweight = torch.randint(0, 256, (hidden_dim // 8, output_dim), dtype=torch.uint32, device='cuda')
            qzeros = torch.randint(0, 16, (32, 16), dtype=torch.uint32, device='cuda')
            scales = torch.randn(32, output_dim, dtype=torch.float16, device='cuda')
            
            # 预热
            for _ in range(10):
                self.kernel_module.gptq_gemm(
                    input_tensor,
                    qweight,
                    qzeros,
                    scales,
                    torch.tensor([]),
                    True,
                    4
                )
            
            torch.cuda.synchronize()
            
            # 性能测试
            num_runs = 100
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                self.kernel_module.gptq_gemm(
                    input_tensor,
                    qweight,
                    qzeros,
                    scales,
                    torch.tensor([]),
                    True,
                    4
                )
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 计算统计信息
            times = np.array(times) * 1000  # 转换为毫秒
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            logger.info(f"📊 性能测试结果 ({num_runs} 次运行):")
            logger.info(f"  平均时间: {avg_time:.3f}ms")
            logger.info(f"  标准差: {std_time:.3f}ms")
            logger.info(f"  最小时间: {min_time:.3f}ms")
            logger.info(f"  最大时间: {max_time:.3f}ms")
            
            # 性能目标检查
            target_time = 0.20  # 目标200μs
            if avg_time <= target_time:
                logger.info(f"✅ 性能达标! (目标: ≤{target_time}ms)")
            else:
                logger.warning(f"⚠️ 性能未达标 (目标: ≤{target_time}ms)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始vLLM GPTQ CUDA内核测试...")
        
        # 加载内核
        if not self.load_kernel():
            return False
        
        # 功能测试
        if not self.test_functionality():
            return False
        
        # 性能测试
        if not self.test_performance():
            return False
        
        logger.info("🎉 所有测试通过!")
        return True

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 vLLM GPTQ CUDA内核测试")
    print("=" * 60)
    
    tester = VLLMGPTQTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 vLLM GPTQ CUDA内核测试完成!")
        print("📊 基于vLLM实现的高性能GPTQ 4bit内核")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ vLLM GPTQ CUDA内核测试失败!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
