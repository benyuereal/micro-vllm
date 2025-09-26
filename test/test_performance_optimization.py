#!/usr/bin/env python3
"""
性能优化测试脚本
测试格式缓存和groupsize优化的效果
"""

import torch
import sys
import os
import logging
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

logging.basicConfig(level=logging.WARNING)  # 减少日志输出
logger = logging.getLogger(__name__)

def test_performance_optimization():
    """测试性能优化效果"""
    print("🚀 性能优化测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试。")
        return
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    print("✅ GPTQ CUDA融合实例创建成功")
    
    # 模拟实际推理数据
    M, K, N = 1, 4096, 12288
    
    # 输入数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 模拟实际推理中的GPTQ格式
    qzeros_actual = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales_actual = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
    qweight_actual = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}")
    print(f"  qweight: {qweight_actual.shape}")
    print(f"  qzeros: {qzeros_actual.shape}")
    print(f"  scales: {scales_actual.shape}")
    
    # 预热
    print("\n🔥 预热阶段...")
    for i in range(5):
        _ = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_actual,
            qzeros=qzeros_actual,
            scales=scales_actual
        )
    
    # 性能测试
    print("\n⚡ 性能测试...")
    times = []
    num_runs = 100
    
    for i in range(num_runs):
        start_time = time.time()
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_actual,
            qzeros=qzeros_actual,
            scales=scales_actual
        )
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        if i % 20 == 0:
            print(f"  迭代 {i}: {times[-1]:.3f}ms")
    
    # 统计结果
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 性能统计:")
    print(f"  平均时间: {avg_time:.3f}ms")
    print(f"  最小时间: {min_time:.3f}ms")
    print(f"  最大时间: {max_time:.3f}ms")
    print(f"  目标达成: {'✅' if avg_time < 0.20 else '❌'} (目标: 0.20ms)")
    
    # 检查格式缓存
    print(f"\n📊 格式缓存状态:")
    print(f"  缓存条目数: {len(gptq_fusion._format_cache)}")
    for key, (qweight, groupsize) in gptq_fusion._format_cache.items():
        print(f"  {key}: groupsize={groupsize}")
    
    return avg_time < 0.20

def test_different_groupsizes():
    """测试不同groupsize的性能"""
    print("\n🧪 测试不同groupsize的性能")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试。")
        return
    
    # 测试不同的groupsize
    groupsizes = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]
    M, K, N = 1, 4096, 12288
    
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    results = {}
    
    for groupsize in groupsizes:
        print(f"\n🔍 测试 groupsize={groupsize}")
        
        # 创建对应的测试数据
        num_groups = K // groupsize
        qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        qweight = torch.randint(0, 256, (K // 8, N), dtype=torch.uint32, device='cuda')
        
        # 创建GPTQ融合实例
        gptq_fusion = GPTQCUDAFusion(groupsize=groupsize)
        
        # 预热
        for _ in range(5):
            _ = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
        
        # 性能测试
        times = []
        for _ in range(20):
            start_time = time.time()
            _ = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_tensor,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        results[groupsize] = avg_time
        print(f"  平均时间: {avg_time:.3f}ms")
    
    # 找出最佳groupsize
    best_groupsize = min(results, key=results.get)
    best_time = results[best_groupsize]
    
    print(f"\n🏆 最佳性能:")
    print(f"  groupsize: {best_groupsize}")
    print(f"  时间: {best_time:.3f}ms")
    
    return best_groupsize, best_time

if __name__ == "__main__":
    print("🚀 开始性能优化测试")
    
    # 测试1: 性能优化
    success1 = test_performance_optimization()
    
    # 测试2: 不同groupsize
    best_groupsize, best_time = test_different_groupsizes()
    
    print(f"\n🎉 性能优化测试完成!")
    print(f"✅ 格式缓存优化: {'通过' if success1 else '失败'}")
    print(f"🏆 最佳groupsize: {best_groupsize} ({best_time:.3f}ms)")
    
    if success1 and best_time < 0.20:
        print("🎯 性能目标达成!")
    else:
        print("⚠️ 性能需要进一步优化")
