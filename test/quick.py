#!/usr/bin/env python3
"""
简单测试验证API服务器格式修复
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def quick_test():
    """快速测试"""
    print("🚀 快速测试API服务器格式修复")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    
    # 模拟API服务器中的实际数据
    M, K, N = 1, 4096, 12288
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 使用错误日志中的确切格式
    qzeros = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
    qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 测试数据:")
    print(f"  qzeros: {qzeros.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  qweight: {qweight.shape}")
    
    try:
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 测试通过! 输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("🎯 修复成功!")
    else:
        print("⚠️ 修复失败")
