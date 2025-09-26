#!/usr/bin/env python3
"""
测试数据类型修复
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def test_dtype_fix():
    """测试数据类型修复"""
    print("🚀 测试数据类型修复")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 使用float16
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')  # 🔧 使用float16
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # 🔧 使用float16
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 测试通过! 输出形状: {output.shape}, dtype: {output.dtype}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_bfloat16_error():
    """测试bfloat16错误情况"""
    print("\n🚀 测试bfloat16错误情况")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 使用bfloat16（应该失败）
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')  # ❌ 使用bfloat16
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.bfloat16, device='cuda')  # ❌ 使用bfloat16
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"❌ 意外成功! 输出形状: {output.shape}, dtype: {output.dtype}")
        return False
    except Exception as e:
        print(f"✅ 预期失败: {e}")
        return True

if __name__ == "__main__":
    print("=" * 50)
    print("🔧 数据类型修复测试")
    print("=" * 50)
    
    # 测试float16（应该成功）
    success1 = test_dtype_fix()
    
    # 测试bfloat16（应该失败）
    success2 = test_bfloat16_error()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎯 数据类型修复测试通过!")
        print("✅ float16测试成功")
        print("✅ bfloat16测试正确失败")
    else:
        print("⚠️ 数据类型修复测试失败")
        print(f"float16测试: {'✅' if success1 else '❌'}")
        print(f"bfloat16测试: {'✅' if success2 else '❌'}")
    print("=" * 50)
