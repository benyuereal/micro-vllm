#!/usr/bin/env python3
"""
测试GPTQ维度验证
验证所有GPTQ参数的维度是否正确
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_gptq_dimensions():
    """测试GPTQ维度验证"""
    print("🚀 GPTQ维度验证测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    try:
        from core.layer.gptq import GPTQCUDAFusion
        
        # 创建GPTQ融合实例
        gptq_fusion = GPTQCUDAFusion(groupsize=128)
        print("✅ GPTQ CUDA融合实例创建成功")
        
        # 测试数据
        M, K, N = 1, 4096, 12288
        groupsize = 128
        num_groups = K // groupsize
        
        input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
        
        print(f"\n📊 测试参数:")
        print(f"  M (batch_size): {M}")
        print(f"  K (input_dim): {K}")
        print(f"  N (output_dim): {N}")
        print(f"  groupsize: {groupsize}")
        print(f"  num_groups: {num_groups}")
        print(f"  K//8: {K//8}")
        
        # 测试1: 正确的维度
        print("\n🧪 测试1: 正确的维度")
        qweight_correct = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
        qzeros_correct = torch.randint(0, 16, (32, 512), dtype=torch.uint32, device='cuda')
        scales_correct = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
        
        print(f"📊 正确维度: qweight{qweight_correct.shape}, qzeros{qzeros_correct.shape}, scales{scales_correct.shape}")
        
        try:
            output1 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_correct, qzeros_correct, scales_correct
            )
            print(f"✅ 正确维度测试成功: {output1.shape}")
        except Exception as e:
            print(f"❌ 正确维度测试失败: {e}")
            return False
        
        # 测试2: 错误的qzeros维度
        print("\n🧪 测试2: 错误的qzeros维度")
        qweight_wrong = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
        qzeros_wrong = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')  # 错误的维度
        scales_wrong = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
        
        print(f"📊 错误qzeros维度: qweight{qweight_wrong.shape}, qzeros{qzeros_wrong.shape}, scales{scales_wrong.shape}")
        
        try:
            output2 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_wrong, qzeros_wrong, scales_wrong
            )
            print(f"❌ 错误维度测试应该失败，但成功了: {output2.shape}")
            return False
        except Exception as e:
            print(f"✅ 错误维度测试正确失败: {e}")
        
        # 测试3: 错误的scales维度
        print("\n🧪 测试3: 错误的scales维度")
        qweight_wrong2 = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
        qzeros_wrong2 = torch.randint(0, 16, (32, 512), dtype=torch.uint32, device='cuda')
        scales_wrong2 = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # 错误的维度
        
        print(f"📊 错误scales维度: qweight{qweight_wrong2.shape}, qzeros{qzeros_wrong2.shape}, scales{scales_wrong2.shape}")
        
        try:
            output3 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_wrong2, qzeros_wrong2, scales_wrong2
            )
            print(f"❌ 错误维度测试应该失败，但成功了: {output3.shape}")
            return False
        except Exception as e:
            print(f"✅ 错误维度测试正确失败: {e}")
        
        print("\n🎉 所有GPTQ维度验证测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_gptq_dimensions()
    if success:
        print("\n🎉 GPTQ维度验证测试通过!")
    else:
        print("\n❌ GPTQ维度验证测试失败!")
        sys.exit(1)
