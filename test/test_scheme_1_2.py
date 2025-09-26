#!/usr/bin/env python3
"""
测试方案1+2：Layer初始化时一次性转换
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def test_scheme_1_2():
    """测试方案1+2"""
    print("🚀 测试方案1+2：Layer初始化时一次性转换")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 使用bfloat16（应该自动转换）
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.bfloat16, device='cuda')
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        # 测试性能
        import time
        
        # 预热
        for _ in range(10):
            gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
        
        torch.cuda.synchronize()
        
        # 性能测试
        timings = []
        for i in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
            end_event.record()
            
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"✅ 方案1+2测试通过!")
        print(f"📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        print(f"  输出形状: {output.shape}, dtype: {output.dtype}")
        
        # 检查是否达到目标
        if avg_time <= 0.12:  # 目标延迟
            print(f"🎯 达到性能目标: {avg_time:.2f}ms <= 0.12ms")
            return True
        else:
            print(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > 0.12ms")
            return False
            
    except Exception as e:
        print(f"❌ 方案1+2测试失败: {e}")
        return False

def test_no_runtime_conversion():
    """测试无运行时转换"""
    print("\n🚀 测试无运行时转换")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 全部float16（无需转换）
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    
    try:
        # 测试性能
        import time
        
        # 预热
        for _ in range(10):
            gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
        
        torch.cuda.synchronize()
        
        # 性能测试
        timings = []
        for i in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
            end_event.record()
            
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"✅ 无运行时转换测试通过!")
        print(f"📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        print(f"  输出形状: {output.shape}, dtype: {output.dtype}")
        
        # 检查是否达到目标
        if avg_time <= 0.10:  # 目标延迟
            print(f"🎯 达到性能目标: {avg_time:.2f}ms <= 0.10ms")
            return True
        else:
            print(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > 0.10ms")
            return False
            
    except Exception as e:
        print(f"❌ 无运行时转换测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 方案1+2：Layer初始化时一次性转换测试")
    print("=" * 60)
    
    # 测试方案1+2
    success1 = test_scheme_1_2()
    
    # 测试无运行时转换
    success2 = test_no_runtime_conversion()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎯 方案1+2测试通过!")
        print("✅ 方案1+2测试成功")
        print("✅ 无运行时转换测试成功")
        print("🎉 性能优化完成!")
    else:
        print("⚠️ 方案1+2测试失败")
        print(f"方案1+2: {'✅' if success1 else '❌'}")
        print(f"无运行时转换: {'✅' if success2 else '❌'}")
    print("=" * 60)
