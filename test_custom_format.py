#!/usr/bin/env python3
"""
测试自定义GPTQ格式
"""
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def test_custom_format():
    """测试自定义GPTQ格式"""
    print("🧪 测试自定义GPTQ格式...")
    
    # 模拟自定义格式的参数
    M, K, N = 8, 4096, 512  # 输入维度4096，输出维度512
    groupsize = 128
    
    # 创建自定义格式的张量
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 自定义格式：qweight [N, custom_cols]
    qweight = torch.randint(0, 256, (N, 64), dtype=torch.int32, device='cuda')  # 64 = N//8
    
    # 自定义格式：scales [num_groups, K]
    num_groups = K // groupsize
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    # 自定义格式：qzeros [num_groups, custom_cols] - 1536列
    qzeros = torch.randint(0, 16, (num_groups, 1536), dtype=torch.int32, device='cuda')
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"qweight形状: {qweight.shape} (自定义格式: [N, 64])")
    print(f"scales形状: {scales.shape} (标准格式: [num_groups, K])")
    print(f"qzeros形状: {qzeros.shape} (自定义格式: [num_groups, 1536])")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试融合算子（应该自动回退到基线实现）
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
    
    test_custom_format()
