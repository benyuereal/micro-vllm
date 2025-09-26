#!/usr/bin/env python3
"""
测试性能优化后的CUDA内核
"""

import torch
import sys
import os
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def test_performance_optimized():
    """测试性能优化后的CUDA内核"""
    print("🚀 测试性能优化后的CUDA内核")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        # 预热
        print("🔥 预热中...")
        for _ in range(10):
            gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
        
        torch.cuda.synchronize()
        
        # 性能测试
        print("⚡ 性能测试中...")
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
        
        print(f"✅ 性能测试完成!")
        print(f"📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        print(f"  输出形状: {output.shape}, dtype: {output.dtype}")
        
        # 检查是否达到目标
        if avg_time <= 0.20:  # 目标延迟
            print(f"🎯 达到性能目标: {avg_time:.2f}ms <= 0.20ms")
            return True
        else:
            print(f"⚠️ 未达到性能目标: {avg_time:.2f}ms > 0.20ms")
            return False
            
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def test_cache_effectiveness():
    """测试缓存有效性"""
    print("\n🚀 测试缓存有效性")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    try:
        # 第一次调用（应该检测格式）
        print("📊 第一次调用（格式检测）...")
        start_time = time.time()
        output1 = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        first_call_time = time.time() - start_time
        
        # 第二次调用（应该使用缓存）
        print("📊 第二次调用（使用缓存）...")
        start_time = time.time()
        output2 = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        second_call_time = time.time() - start_time
        
        print(f"✅ 缓存测试完成!")
        print(f"📊 缓存效果:")
        print(f"  第一次调用: {first_call_time*1000:.2f}ms")
        print(f"  第二次调用: {second_call_time*1000:.2f}ms")
        print(f"  缓存加速: {first_call_time/second_call_time:.2f}x")
        
        # 检查输出一致性
        if torch.allclose(output1, output2, atol=1e-3):
            print("✅ 输出一致性检查通过")
            return True
        else:
            print("❌ 输出一致性检查失败")
            return False
            
    except Exception as e:
        print(f"❌ 缓存测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 性能优化测试")
    print("=" * 60)
    
    # 测试性能优化
    success1 = test_performance_optimized()
    
    # 测试缓存有效性
    success2 = test_cache_effectiveness()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎯 性能优化测试通过!")
        print("✅ 性能优化成功")
        print("✅ 缓存有效性测试通过")
        print("🎉 性能优化完成!")
    else:
        print("⚠️ 性能优化测试失败")
        print(f"性能优化: {'✅' if success1 else '❌'}")
        print(f"缓存有效性: {'✅' if success2 else '❌'}")
    print("=" * 60)
