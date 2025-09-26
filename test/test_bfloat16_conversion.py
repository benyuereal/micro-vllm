#!/usr/bin/env python3
"""
测试bfloat16到float16自动转换修复
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def test_bfloat16_conversion():
    """测试bfloat16自动转换"""
    print("🚀 测试bfloat16自动转换")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 使用bfloat16（应该自动转换）
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')  # ❌ 使用bfloat16
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.bfloat16, device='cuda')  # ❌ 使用bfloat16
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 自动转换测试通过! 输出形状: {output.shape}, dtype: {output.dtype}")
        return True
    except Exception as e:
        print(f"❌ 自动转换测试失败: {e}")
        return False

def test_mixed_dtypes():
    """测试混合数据类型"""
    print("\n🚀 测试混合数据类型")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 混合数据类型
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')  # ✅ float16
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.bfloat16, device='cuda')  # ❌ bfloat16
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 混合数据类型测试通过! 输出形状: {output.shape}, dtype: {output.dtype}")
        return True
    except Exception as e:
        print(f"❌ 混合数据类型测试失败: {e}")
        return False

def test_all_float16():
    """测试全部float16"""
    print("\n🚀 测试全部float16")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟实际数据 - 全部float16
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')  # ✅ float16
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # ✅ float16
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  input: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
    print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
    print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 全部float16测试通过! 输出形状: {output.shape}, dtype: {output.dtype}")
        return True
    except Exception as e:
        print(f"❌ 全部float16测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 bfloat16自动转换修复测试")
    print("=" * 60)
    
    # 测试bfloat16自动转换
    success1 = test_bfloat16_conversion()
    
    # 测试混合数据类型
    success2 = test_mixed_dtypes()
    
    # 测试全部float16
    success3 = test_all_float16()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("🎯 bfloat16自动转换修复测试通过!")
        print("✅ bfloat16自动转换测试成功")
        print("✅ 混合数据类型测试成功")
        print("✅ 全部float16测试成功")
    else:
        print("⚠️ bfloat16自动转换修复测试失败")
        print(f"bfloat16转换: {'✅' if success1 else '❌'}")
        print(f"混合数据类型: {'✅' if success2 else '❌'}")
        print(f"全部float16: {'✅' if success3 else '❌'}")
    print("=" * 60)
