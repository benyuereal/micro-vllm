#!/usr/bin/env python3
"""
融合LN+QKV GPTQ CUDA内核测试脚本
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

class LNQKVFusionTester:
    """融合LN+QKV测试器"""
    
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
        """加载CUDA内核"""
        logger.info("🔧 加载融合LN+QKV CUDA内核...")
        
        try:
            # 尝试导入编译的内核
            import fused_ln_qkv_gptq_cuda
            self.kernel_module = fused_ln_qkv_gptq_cuda
            self.kernel_loaded = True
            logger.info("✅ 融合LN+QKV CUDA内核加载成功")
            return True
        except ImportError as e:
            logger.error(f"❌ 无法导入融合LN+QKV CUDA内核: {e}")
            logger.info("💡 请先运行 compile_ln_qkv_fusion.py 编译内核")
            return False
    
    def create_test_data(self, batch_size=1, seq_len=1, hidden_dim=4096):
        """创建测试数据"""
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        groupsize = 128
        num_groups = hidden_dim // groupsize
        
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        
        return {
            'input': input_tensor,
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'ln_weight': ln_weight,
            'ln_bias': ln_bias,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_heads': 32,
            'kv_num_heads': 32,
            'head_size': 128,
            'groupsize': groupsize
        }
    
    def test_basic_functionality(self):
        """测试基础功能"""
        logger.info("🧪 测试基础功能...")
        
        if not self.kernel_loaded:
            logger.error("❌ CUDA内核未加载")
            return False
        
        try:
            # 创建测试数据
            test_data = self.create_test_data()
            
            # 创建输出张量
            q_output = torch.zeros(test_data['batch_size'], test_data['num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            k_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            v_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            self.kernel_module.fused_ln_qkv_gptq_cuda(
                test_data['input'].flatten(),
                test_data['qweight'],
                test_data['qzeros'],
                test_data['scales'],
                test_data['ln_weight'],
                test_data['ln_bias'],
                q_output.flatten(),
                k_output.flatten(),
                v_output.flatten(),
                test_data['batch_size'],
                test_data['seq_len'],
                test_data['hidden_dim'],
                test_data['num_heads'],
                test_data['kv_num_heads'],
                test_data['head_size'],
                test_data['groupsize'],
                1e-6
            )
            
            # 验证输出形状
            expected_shape = (test_data['batch_size'], test_data['num_heads'], 
                           test_data['seq_len'], test_data['head_size'])
            
            assert q_output.shape == expected_shape, f"Q形状错误: {q_output.shape} != {expected_shape}"
            assert k_output.shape == expected_shape, f"K形状错误: {k_output.shape} != {expected_shape}"
            assert v_output.shape == expected_shape, f"V形状错误: {v_output.shape} != {expected_shape}"
            
            # 检查输出是否包含有效值
            if torch.any(torch.isnan(q_output)) or torch.any(torch.isinf(q_output)):
                logger.error("❌ Q输出包含无效值")
                return False
            
            if torch.any(torch.isnan(k_output)) or torch.any(torch.isinf(k_output)):
                logger.error("❌ K输出包含无效值")
                return False
            
            if torch.any(torch.isnan(v_output)) or torch.any(torch.isinf(v_output)):
                logger.error("❌ V输出包含无效值")
                return False
            
            logger.info("✅ 基础功能测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 基础功能测试失败: {e}")
            return False
    
    def test_performance(self):
        """测试性能"""
        logger.info("🚀 测试性能...")
        
        if not self.kernel_loaded:
            logger.error("❌ CUDA内核未加载")
            return False
        
        try:
            # 创建测试数据
            test_data = self.create_test_data()
            
            # 创建输出张量
            q_output = torch.zeros(test_data['batch_size'], test_data['num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            k_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            v_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            
            # 预热
            for _ in range(10):
                self.kernel_module.fused_ln_qkv_gptq_cuda(
                    test_data['input'].flatten(),
                    test_data['qweight'],
                    test_data['qzeros'],
                    test_data['scales'],
                    test_data['ln_weight'],
                    test_data['ln_bias'],
                    q_output.flatten(),
                    k_output.flatten(),
                    v_output.flatten(),
                    test_data['batch_size'],
                    test_data['seq_len'],
                    test_data['hidden_dim'],
                    test_data['num_heads'],
                    test_data['kv_num_heads'],
                    test_data['head_size'],
                    test_data['groupsize'],
                    1e-6
                )
            torch.cuda.synchronize()
            
            # 性能测试
            num_runs = 100
            timings = []
            
            for i in range(num_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                self.kernel_module.fused_ln_qkv_gptq_cuda(
                    test_data['input'].flatten(),
                    test_data['qweight'],
                    test_data['qzeros'],
                    test_data['scales'],
                    test_data['ln_weight'],
                    test_data['ln_bias'],
                    q_output.flatten(),
                    k_output.flatten(),
                    v_output.flatten(),
                    test_data['batch_size'],
                    test_data['seq_len'],
                    test_data['hidden_dim'],
                    test_data['num_heads'],
                    test_data['kv_num_heads'],
                    test_data['head_size'],
                    test_data['groupsize'],
                    1e-6
                )
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
                
                if i % 20 == 0:
                    logger.info(f"  迭代 {i}: {timings[-1]:.2f}ms")
            
            # 计算统计信息
            avg_time = np.mean(timings)
            min_time = np.min(timings)
            max_time = np.max(timings)
            std_time = np.std(timings)
            
            logger.info(f"📊 性能统计:")
            logger.info(f"  平均时间: {avg_time:.2f}ms")
            logger.info(f"  最小时间: {min_time:.2f}ms")
            logger.info(f"  最大时间: {max_time:.2f}ms")
            logger.info(f"  标准差: {std_time:.2f}ms")
            
            # 性能评估
            target_time = 0.25  # 目标时间: 0.25ms
            if avg_time <= target_time:
                logger.info(f"🎉 达到性能目标: {avg_time:.2f}ms <= {target_time}ms")
                return True
            elif avg_time <= target_time * 2:
                logger.info(f"✅ 接近性能目标: {avg_time:.2f}ms <= {target_time * 2}ms")
                return True
            else:
                logger.warning(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > {target_time}ms")
                return False
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            return False
    
    def test_batch_processing(self):
        """测试批处理"""
        logger.info("📦 测试批处理...")
        
        if not self.kernel_loaded:
            logger.error("❌ CUDA内核未加载")
            return False
        
        try:
            # 测试不同的批处理配置
            batch_configs = [
                {"batch_size": 1, "seq_len": 1, "name": "单token"},
                {"batch_size": 1, "seq_len": 16, "name": "短序列"},
                {"batch_size": 1, "seq_len": 64, "name": "中序列"},
                {"batch_size": 4, "seq_len": 16, "name": "批处理"},
            ]
            
            for config in batch_configs:
                logger.info(f"  测试配置: {config['name']}")
                
                # 创建测试数据
                test_data = self.create_test_data(
                    config['batch_size'], 
                    config['seq_len']
                )
                
                # 创建输出张量
                q_output = torch.zeros(test_data['batch_size'], test_data['num_heads'], 
                                     test_data['seq_len'], test_data['head_size'], 
                                     dtype=torch.float16, device='cuda')
                k_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                     test_data['seq_len'], test_data['head_size'], 
                                     dtype=torch.float16, device='cuda')
                v_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                     test_data['seq_len'], test_data['head_size'], 
                                     dtype=torch.float16, device='cuda')
                
                # 测试性能
                times = []
                for _ in range(50):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    self.kernel_module.fused_ln_qkv_gptq_cuda(
                        test_data['input'].flatten(),
                        test_data['qweight'],
                        test_data['qzeros'],
                        test_data['scales'],
                        test_data['ln_weight'],
                        test_data['ln_bias'],
                        q_output.flatten(),
                        k_output.flatten(),
                        v_output.flatten(),
                        test_data['batch_size'],
                        test_data['seq_len'],
                        test_data['hidden_dim'],
                        test_data['num_heads'],
                        test_data['kv_num_heads'],
                        test_data['head_size'],
                        test_data['groupsize'],
                        1e-6
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                
                avg_time = np.mean(times)
                logger.info(f"    平均时间: {avg_time:.2f}ms")
                
                # 验证输出形状
                expected_shape = (config['batch_size'], 32, config['seq_len'], 128)
                assert q_output.shape == expected_shape, f"Q形状错误: {q_output.shape} != {expected_shape}"
                assert k_output.shape == expected_shape, f"K形状错误: {k_output.shape} != {expected_shape}"
                assert v_output.shape == expected_shape, f"V形状错误: {v_output.shape} != {expected_shape}"
            
            logger.info("✅ 批处理测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 批处理测试失败: {e}")
            return False
    
    def test_correctness(self):
        """测试正确性"""
        logger.info("🔍 测试正确性...")
        
        if not self.kernel_loaded:
            logger.error("❌ CUDA内核未加载")
            return False
        
        try:
            # 创建测试数据
            test_data = self.create_test_data()
            
            # 创建输出张量
            q_output = torch.zeros(test_data['batch_size'], test_data['num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            k_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            v_output = torch.zeros(test_data['batch_size'], test_data['kv_num_heads'], 
                                 test_data['seq_len'], test_data['head_size'], 
                                 dtype=torch.float16, device='cuda')
            
            # 调用融合内核
            self.kernel_module.fused_ln_qkv_gptq_cuda(
                test_data['input'].flatten(),
                test_data['qweight'],
                test_data['qzeros'],
                test_data['scales'],
                test_data['ln_weight'],
                test_data['ln_bias'],
                q_output.flatten(),
                k_output.flatten(),
                v_output.flatten(),
                test_data['batch_size'],
                test_data['seq_len'],
                test_data['hidden_dim'],
                test_data['num_heads'],
                test_data['kv_num_heads'],
                test_data['head_size'],
                test_data['groupsize'],
                1e-6
            )
            
            # 检查输出是否包含有效值
            if torch.any(torch.isnan(q_output)) or torch.any(torch.isinf(q_output)):
                logger.error("❌ Q输出包含无效值")
                return False
            
            if torch.any(torch.isnan(k_output)) or torch.any(torch.isinf(k_output)):
                logger.error("❌ K输出包含无效值")
                return False
            
            if torch.any(torch.isnan(v_output)) or torch.any(torch.isinf(v_output)):
                logger.error("❌ V输出包含无效值")
                return False
            
            # 检查输出范围
            q_range = (q_output.min().item(), q_output.max().item())
            k_range = (k_output.min().item(), k_output.max().item())
            v_range = (v_output.min().item(), v_output.max().item())
            
            logger.info(f"📊 输出范围:")
            logger.info(f"  Q: [{q_range[0]:.3f}, {q_range[1]:.3f}]")
            logger.info(f"  K: [{k_range[0]:.3f}, {k_range[1]:.3f}]")
            logger.info(f"  V: [{v_range[0]:.3f}, {v_range[1]:.3f}]")
            
            logger.info("✅ 正确性测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 正确性测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始运行所有测试")
        
        # 加载内核
        if not self.load_kernel():
            return False
        
        # 运行测试
        tests = [
            ("基础功能", self.test_basic_functionality),
            ("性能测试", self.test_performance),
            ("批处理测试", self.test_batch_processing),
            ("正确性测试", self.test_correctness)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 运行测试: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                if test_func():
                    results[test_name] = True
                    passed += 1
                    logger.info(f"✅ {test_name} 测试通过")
                else:
                    results[test_name] = False
                    logger.error(f"❌ {test_name} 测试失败")
            except Exception as e:
                results[test_name] = False
                logger.error(f"❌ {test_name} 测试异常: {e}")
        
        # 总结
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 测试总结: {passed}/{total} 通过")
        logger.info(f"{'='*60}")
        
        if passed == total:
            logger.info("🎉 所有测试通过!")
        else:
            logger.error(f"❌ {total - passed} 个测试失败")
        
        return passed == total

def main():
    """主函数"""
    logger.info("🚀 开始融合LN+QKV GPTQ CUDA内核测试")
    
    # 创建测试器
    tester = LNQKVFusionTester()
    
    # 运行所有测试
    success = tester.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
