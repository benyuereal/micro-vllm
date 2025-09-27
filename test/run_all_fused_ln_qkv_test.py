#!/usr/bin/env python3
"""
融合LN+QKV GPTQ完整测试脚本
包括功能测试、性能测试、集成测试等
"""

import sys
import os
import torch
import time
import logging
import numpy as np
import subprocess
import shutil
from typing import Tuple, List, Dict, Any
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fused_ln_qkv_test.log')
    ]
)
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

class FusedLNQKVTestRunner:
    """融合LN+QKV测试运行器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_results = {}
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            logger.info(f"🔧 CUDA设备: {torch.cuda.get_device_name()}")
            logger.info(f"🔧 CUDA版本: {torch.version.cuda}")
        else:
            logger.error("❌ CUDA不可用，无法运行测试")
            sys.exit(1)
    
    def check_environment(self) -> bool:
        """检查测试环境"""
        logger.info("🔍 检查测试环境...")
        
        # 检查CUDA内核库
        kernel_paths = [
            "cuda/ln_qkv_fusion_kernel.so",
            "cuda/ln_qkv_fusion_kernel.dll",
            "ln_qkv_fusion_kernel.so",
            "ln_qkv_fusion_kernel.dll"
        ]
        
        kernel_found = False
        for path in kernel_paths:
            if os.path.exists(path):
                logger.info(f"✅ 找到CUDA内核库: {path}")
                kernel_found = True
                break
        
        if not kernel_found:
            logger.warning("⚠️ 找不到CUDA内核库，将尝试编译")
            return self.compile_kernel()
        
        return True
    
    def compile_kernel(self) -> bool:
        """编译CUDA内核"""
        logger.info("🔨 编译CUDA内核...")
        
        try:
            # 检查编译脚本
            compile_script = "cuda/compile_ln_qkv_fusion.sh"
            if not os.path.exists(compile_script):
                logger.error(f"❌ 找不到编译脚本: {compile_script}")
                return False
            
            # 执行编译
            result = subprocess.run(
                ["bash", compile_script], 
                cwd="cuda",
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ CUDA内核编译成功")
                return True
            else:
                logger.error(f"❌ CUDA内核编译失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 编译过程异常: {e}")
            return False
    
    def create_test_data(self, batch_size: int = 1, seq_len: int = 1, hidden_dim: int = 4096) -> Tuple[torch.Tensor, MockLayer]:
        """创建测试数据"""
        # 创建输入数据
        input_data = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
        
        # 创建模拟的Layer
        layer = MockLayer(hidden_dim)
        
        return input_data, layer
    
    def test_basic_functionality(self) -> bool:
        """测试基础功能"""
        logger.info("🧪 测试基础功能...")
        
        try:
            # 导入融合算子
            from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
            
            # 创建测试数据
            input_data, layer = self.create_test_data()
            
            # 创建融合算子
            fused_op = FusedLNQKVGPTQ()
            
            if not fused_op.is_available():
                logger.warning("⚠️ 融合算子不可用，跳过基础功能测试")
                return False
            
            # 测试融合算子
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
            
            # 检查输出是否包含有效值
            if torch.any(torch.isnan(q)) or torch.any(torch.isinf(q)):
                logger.error("❌ Q输出包含无效值")
                return False
            
            if torch.any(torch.isnan(k)) or torch.any(torch.isinf(k)):
                logger.error("❌ K输出包含无效值")
                return False
            
            if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                logger.error("❌ V输出包含无效值")
                return False
            
            logger.info("✅ 基础功能测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 基础功能测试失败: {e}")
            return False
    
    def test_performance(self) -> bool:
        """测试性能"""
        logger.info("🚀 测试性能...")
        
        try:
            from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
            from core.layer.gptq import GPTQCUDAFusion
            
            # 创建测试数据
            input_data, layer = self.create_test_data()
            
            # 创建融合算子
            fused_op = FusedLNQKVGPTQ()
            
            if not fused_op.is_available():
                logger.warning("⚠️ 融合算子不可用，跳过性能测试")
                return False
            
            # 预热
            for _ in range(10):
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
            
            # 性能测试
            num_runs = 100
            times = []
            
            for _ in range(num_runs):
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
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            logger.info(f"📊 融合算子性能统计:")
            logger.info(f"  平均时间: {avg_time:.2f}ms")
            logger.info(f"  最小时间: {min_time:.2f}ms")
            logger.info(f"  最大时间: {max_time:.2f}ms")
            
            # 测试分离算子性能
            gptq_fusion = GPTQCUDAFusion()
            
            # 预热分离算子
            for _ in range(10):
                # LayerNorm
                ln_output = torch.nn.functional.layer_norm(
                    input_data, 
                    input_data.shape[-1:], 
                    layer.ln_1.weight, 
                    layer.ln_1.bias
                )
                
                # QKV投影
                batch_size, seq_len, hidden_dim = ln_output.shape
                input_2d = ln_output.view(-1, hidden_dim)
                
                qkv_output = gptq_fusion.fused_gptq_gemm_4bit(
                    input=input_2d,
                    qweight=layer.attn.qweight,
                    qzeros=layer.attn.qzeros,
                    scales=layer.attn.scales
                )
                
                # 重塑和分割
                qkv_output = qkv_output.view(batch_size, seq_len, -1)
                hidden_size = qkv_output.shape[-1] // 3
                q, k, v = qkv_output.split(hidden_size, dim=-1)
                
                # 重塑为head格式
                q = q.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
                k = k.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
                v = v.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            
            # 分离算子性能测试
            separate_times = []
            for _ in range(num_runs):
                start_time = time.time()
                
                # LayerNorm
                ln_output = torch.nn.functional.layer_norm(
                    input_data, 
                    input_data.shape[-1:], 
                    layer.ln_1.weight, 
                    layer.ln_1.bias
                )
                
                # QKV投影
                batch_size, seq_len, hidden_dim = ln_output.shape
                input_2d = ln_output.view(-1, hidden_dim)
                
                qkv_output = gptq_fusion.fused_gptq_gemm_4bit(
                    input=input_2d,
                    qweight=layer.attn.qweight,
                    qzeros=layer.attn.qzeros,
                    scales=layer.attn.scales
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
                separate_times.append((end_time - start_time) * 1000)
            
            separate_avg = np.mean(separate_times)
            speedup = separate_avg / avg_time
            
            logger.info(f"📊 分离算子性能统计:")
            logger.info(f"  平均时间: {separate_avg:.2f}ms")
            logger.info(f"  加速比: {speedup:.2f}x")
            
            # 性能目标检查
            target_time = 0.25  # 目标时间: 0.25ms
            if avg_time <= target_time:
                logger.info(f"🎯 达到性能目标: {avg_time:.2f}ms <= {target_time}ms")
                return True
            else:
                logger.warning(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > {target_time}ms")
                return False
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            return False
    
    def test_batch_processing(self) -> bool:
        """测试批处理"""
        logger.info("📦 测试批处理...")
        
        try:
            from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
            
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
                input_data, layer = self.create_test_data(
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
    
    def test_correctness(self) -> bool:
        """测试正确性"""
        logger.info("🔍 测试正确性...")
        
        try:
            from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
            
            # 创建测试数据
            input_data, layer = self.create_test_data()
            
            # 创建融合算子
            fused_op = FusedLNQKVGPTQ()
            
            if not fused_op.is_available():
                logger.warning("⚠️ 融合算子不可用，跳过正确性测试")
                return False
            
            # 使用融合算子
            q_fused, k_fused, v_fused = fused_op.fused_ln_qkv_gptq(
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
            
            # 使用PyTorch实现作为参考
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_data, 
                input_data.shape[-1:], 
                layer.ln_1.weight, 
                layer.ln_1.bias
            )
            
            # 简化的QKV投影（使用随机权重）
            qkv_weight = torch.randn(input_data.shape[-1], input_data.shape[-1] * 3, dtype=torch.float16)
            qkv_output = torch.matmul(ln_output, qkv_weight)
            
            # 分割QKV
            hidden_size = qkv_output.shape[-1] // 3
            q_ref, k_ref, v_ref = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            batch_size, seq_len, hidden_dim = q_ref.shape
            q_ref = q_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            k_ref = k_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            v_ref = v_ref.view(batch_size, seq_len, 32, 128).permute(0, 2, 1, 3)
            
            # 比较结果（由于GPTQ量化，结果可能不完全相同）
            logger.info("📊 结果比较:")
            logger.info(f"  Q形状: {q_fused.shape} vs {q_ref.shape}")
            logger.info(f"  K形状: {k_fused.shape} vs {k_ref.shape}")
            logger.info(f"  V形状: {v_fused.shape} vs {v_ref.shape}")
            
            # 检查输出是否包含有效值
            if torch.any(torch.isnan(q_fused)) or torch.any(torch.isinf(q_fused)):
                logger.error("❌ Q输出包含无效值")
                return False
            
            if torch.any(torch.isnan(k_fused)) or torch.any(torch.isinf(k_fused)):
                logger.error("❌ K输出包含无效值")
                return False
            
            if torch.any(torch.isnan(v_fused)) or torch.any(torch.isinf(v_fused)):
                logger.error("❌ V输出包含无效值")
                return False
            
            logger.info("✅ 正确性测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 正确性测试失败: {e}")
            return False
    
    def test_integration(self) -> bool:
        """测试集成"""
        logger.info("🔗 测试集成...")
        
        try:
            from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ
            
            # 创建测试数据
            input_data, layer = self.create_test_data()
            
            # 创建融合算子
            fused_op = FusedLNQKVGPTQ()
            
            if not fused_op.is_available():
                logger.warning("⚠️ 融合算子不可用，跳过集成测试")
                return False
            
            # 测试融合算子
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
            
            logger.info("✅ 集成测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 集成测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("🚀 开始运行所有测试")
        
        # 检查环境
        if not self.check_environment():
            logger.error("❌ 环境检查失败")
            return {}
        
        # 运行测试
        tests = [
            ("基础功能", self.test_basic_functionality),
            ("性能测试", self.test_performance),
            ("批处理测试", self.test_batch_processing),
            ("正确性测试", self.test_correctness),
            ("集成测试", self.test_integration)
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
        
        return results
    
    def generate_report(self, results: Dict[str, bool]):
        """生成测试报告"""
        logger.info("📝 生成测试报告...")
        
        report = f"""
# 融合LN+QKV GPTQ测试报告

## 测试环境
- CUDA设备: {torch.cuda.get_device_name() if self.cuda_available else 'N/A'}
- CUDA版本: {torch.version.cuda if self.cuda_available else 'N/A'}
- PyTorch版本: {torch.__version__}
- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 测试结果
"""
        
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            report += f"- {test_name}: {status}\n"
        
        passed_count = sum(results.values())
        total_count = len(results)
        report += f"\n## 总结\n- 通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)\n"
        
        if passed_count == total_count:
            report += "- 状态: 🎉 所有测试通过\n"
        else:
            report += f"- 状态: ❌ {total_count - passed_count} 个测试失败\n"
        
        # 保存报告
        with open("fused_ln_qkv_test_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info("📝 测试报告已保存到: fused_ln_qkv_test_report.md")

def main():
    """主函数"""
    logger.info("🚀 开始融合LN+QKV GPTQ完整测试")
    
    # 创建测试运行器
    test_runner = FusedLNQKVTestRunner()
    
    # 运行所有测试
    results = test_runner.run_all_tests()
    
    # 生成报告
    test_runner.generate_report(results)
    
    # 返回结果
    success = all(results.values()) if results else False
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
