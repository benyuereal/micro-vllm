#!/usr/bin/env python3
"""
测试实际推理中的GPTQ格式修复
验证qzeros [32, 1536] 和 scales [32, 12288] 格式的处理
"""

import torch
import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_actual_inference_format_fix():
    """测试实际推理中的GPTQ格式修复"""
    print("🚀 测试实际推理中的GPTQ格式修复")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试。")
        return
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    print("✅ GPTQ CUDA融合实例创建成功")
    
    # 模拟实际推理数据 - 使用错误日志中的确切格式
    M, K, N = 1, 4096, 12288
    
    # 输入数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 模拟实际推理中的GPTQ格式 - 使用错误日志中的确切维度
    qzeros_actual = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales_actual = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # 注意：这里是12288而不是4096
    qweight_actual = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 实际推理格式 (来自错误日志):")
    print(f"  input: {input_tensor.shape}")
    print(f"  qweight: {qweight_actual.shape}")
    print(f"  qzeros: {qzeros_actual.shape}")
    print(f"  scales: {scales_actual.shape}")
    
    try:
        # 测试CUDA内核
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_actual,
            qzeros=qzeros_actual,
            scales=scales_actual
        )
        
        print(f"✅ 实际推理格式修复测试通过!")
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 期望形状: torch.Size([{M}, {N}])")
        
        assert output.shape == (M, N), f"输出形状错误: {output.shape}"
        
    except Exception as e:
        print(f"❌ 实际推理格式修复测试失败: {e}")
        return False
    
    # 测试性能
    print("\n⚡ 性能测试...")
    try:
        result = gptq_fusion.benchmark(
            input=input_tensor,
            qweight=qweight_actual,
            qzeros=qzeros_actual,
            scales=scales_actual,
            num_runs=10
        )
        
        print(f"📊 性能统计:")
        print(f"  平均时间: {result['avg_time']:.3f}ms")
        print(f"  最小时间: {result['min_time']:.3f}ms")
        print(f"  最大时间: {result['max_time']:.3f}ms")
        print(f"  目标达成: {'✅' if result['target_achieved'] else '❌'} (目标: 0.20ms)")
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False
    
    print("\n🎉 实际推理格式修复测试完成!")
    return True

def test_format_detection():
    """测试格式检测逻辑"""
    print("\n🔍 测试格式检测逻辑")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试。")
        return
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 测试数据
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 测试1: 标准格式
    print("\n🧪 测试1: 标准格式")
    qzeros_std = torch.randint(0, 16, (32, 512), dtype=torch.uint32, device='cuda')  # [num_groups, K//8]
    scales_std = torch.randn(32, 4096, dtype=torch.float16, device='cuda')  # [num_groups, K]
    qweight_std = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_std,
            qzeros=qzeros_std,
            scales=scales_std
        )
        print(f"✅ 标准格式测试通过! 输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 标准格式测试失败: {e}")
        return False
    
    # 测试2: 实际推理格式
    print("\n🧪 测试2: 实际推理格式")
    qzeros_real = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')  # [num_groups, groupsize//8]
    scales_real = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # [num_groups, N]
    qweight_real = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_real,
            qzeros=qzeros_real,
            scales=scales_real
        )
        print(f"✅ 实际推理格式测试通过! 输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 实际推理格式测试失败: {e}")
        return False
    
    print("\n🎉 格式检测逻辑测试完成!")
    return True

if __name__ == "__main__":
    print("🚀 开始实际推理格式修复测试")
    
    # 测试1: 实际推理格式修复
    success1 = test_actual_inference_format_fix()
    
    # 测试2: 格式检测逻辑
    success2 = test_format_detection()
    
    print(f"\n🎉 实际推理格式修复测试完成!")
    print(f"✅ 实际推理格式修复: {'通过' if success1 else '失败'}")
    print(f"✅ 格式检测逻辑: {'通过' if success2 else '失败'}")
    
    if success1 and success2:
        print("🎯 所有测试通过!")
    else:
        print("⚠️ 部分测试失败")
