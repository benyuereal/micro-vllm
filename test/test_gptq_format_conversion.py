#!/usr/bin/env python3
"""
测试GPTQ格式自动转换
验证CUDA内核能正确处理不同的GPTQ格式
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_gptq_format_conversion():
    """测试GPTQ格式自动转换"""
    print("🚀 GPTQ格式自动转换测试")
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
        
        # 测试1: 标准格式 [N, K//8]
        print("\n🧪 测试1: 标准格式 [N, K//8]")
        qweight_standard = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros_standard = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales_standard = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        print(f"📊 标准格式: qweight{qweight_standard.shape}, qzeros{qzeros_standard.shape}, scales{scales_standard.shape}")
        
        try:
            output1 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_standard, qzeros_standard, scales_standard
            )
            print(f"✅ 标准格式测试成功: {output1.shape}")
        except Exception as e:
            print(f"❌ 标准格式测试失败: {e}")
            return False
        
        # 测试2: 转置格式 [K//8, N]
        print("\n🧪 测试2: 转置格式 [K//8, N]")
        qweight_transposed = torch.randint(0, 256, (K // 8, N), dtype=torch.uint32, device='cuda')
        qzeros_transposed = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales_transposed = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        print(f"📊 转置格式: qweight{qweight_transposed.shape}, qzeros{qzeros_transposed.shape}, scales{scales_transposed.shape}")
        
        try:
            output2 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_transposed, qzeros_transposed, scales_transposed
            )
            print(f"✅ 转置格式测试成功: {output2.shape}")
        except Exception as e:
            print(f"❌ 转置格式测试失败: {e}")
            return False
        
        # 测试3: 实际推理格式 (模拟从日志中看到的格式)
        print("\n🧪 测试3: 实际推理格式")
        # 从日志: qweight512x12288, qzeros[32, 1536], scales[32, 12288]
        # 这意味着: qweight是[K//8, N]格式，需要转换
        qweight_real = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
        qzeros_real = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
        scales_real = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
        
        print(f"📊 实际格式: qweight{qweight_real.shape}, qzeros{qzeros_real.shape}, scales{scales_real.shape}")
        
        try:
            output3 = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight_real, qzeros_real, scales_real
            )
            print(f"✅ 实际格式测试成功: {output3.shape}")
        except Exception as e:
            print(f"❌ 实际格式测试失败: {e}")
            return False
        
        print("\n🎉 所有GPTQ格式转换测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_gptq_format_conversion()
    if success:
        print("\n🎉 GPTQ格式转换测试通过!")
    else:
        print("\n❌ GPTQ格式转换测试失败!")
        sys.exit(1)
