#!/usr/bin/env python3
"""
测试GPTQ格式修复
"""
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def test_gptq_format():
    """测试GPTQ格式是否正确"""
    print("🧪 测试GPTQ格式修复...")
    
    # 模拟GPTQ格式的参数
    M, K, N = 32, 4096, 512  # 输入维度4096，输出维度512
    groupsize = 128
    
    # 创建GPTQ格式的张量
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # GPTQ格式：qweight [N, K//8]
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    
    # GPTQ格式：scales [num_groups, K]
    num_groups = K // groupsize
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    # GPTQ格式：qzeros [num_groups, K//8]
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"qweight形状: {qweight.shape} (GPTQ格式: [N, K//8])")
    print(f"scales形状: {scales.shape} (GPTQ格式: [num_groups, K])")
    print(f"qzeros形状: {qzeros.shape} (GPTQ格式: [num_groups, K//8])")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试融合算子
        print("🚀 测试融合算子...")
        result = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        print(f"✅ 融合算子成功！输出形状: {result.shape}")
        
        # 测试基线实现
        print("🐌 测试基线实现...")
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 基线实现成功！输出形状: {baseline_result.shape}")
        
        # 比较结果
        diff = torch.abs(result - baseline_result).max()
        print(f"📊 最大差异: {diff.item():.6f}")
        
        if diff < 1e-3:
            print("🎉 测试通过！融合算子和基线实现结果一致")
        else:
            print("⚠️  结果有差异，可能需要进一步调试")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        sys.exit(1)
    
    test_gptq_format()
