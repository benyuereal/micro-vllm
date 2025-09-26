#!/usr/bin/env python3
"""
简单的GPTQ功能测试
验证CUDA融合内核基本功能
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def simple_test():
    """简单测试"""
    print("🚀 简单GPTQ功能测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    try:
        from core.layer.gptq import GPTQCUDAFusion
        print("✅ GPTQCUDAFusion导入成功!")
        
        # 创建实例
        gptq_fusion = GPTQCUDAFusion(groupsize=128)
        print("✅ GPTQCUDAFusion实例创建成功!")
        
        # 简单功能测试
        print("\n🧪 简单功能测试...")
        M, K, N = 1, 4096, 12288
        groupsize = 128
        num_groups = K // groupsize
        
        input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        print(f"📊 输入形状: {input_tensor.shape}")
        print(f"📊 qweight形状: {qweight.shape}")
        
        # 执行测试
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input_tensor, qweight, qzeros, scales
        )
        
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 期望形状: torch.Size([1, 12288])")
        
        if output.shape == (1, 12288):
            print("✅ 功能测试成功!")
            return True
        else:
            print("❌ 功能测试失败!")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 简单测试通过!")
    else:
        print("\n❌ 简单测试失败!")
        sys.exit(1)
