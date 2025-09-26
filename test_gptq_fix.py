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
import torch.nn.functional as F

def fast_baseline_gptq_gemm(input, qweight, qzeros, scales, groupsize):
    """
    更快的基线实现：使用PyTorch向量化操作
    但仍然比Triton融合算子慢，作为合理的基准
    """
    N, _ = qweight.shape
    K = scales.shape[1]
    num_groups = scales.shape[0]
    
    # 使用向量化操作进行反量化
    dequantized_weight = torch.zeros((N, K), dtype=torch.float16, device=qweight.device)
    
    for group_idx in range(num_groups):
        start_idx = group_idx * groupsize
        end_idx = min(start_idx + groupsize, K)
        
        # 获取当前组的参数
        group_scales = scales[group_idx, start_idx:end_idx]  # [group_size]
        group_qzeros = qzeros[group_idx, start_idx//8:(end_idx+7)//8]  # [group_size//8]
        
        # 向量化处理当前组
        for k in range(start_idx, end_idx):
            byte_idx = k // 8
            bit_shift = (k % 8) * 4
            
            # 提取所有输出维度的4bit值
            weight_vals = (qweight[:, byte_idx] >> bit_shift) & 0xF  # [N]
            zero_val = (group_qzeros[byte_idx] >> bit_shift) & 0xF
            
            # 反量化
            dequantized_weight[:, k] = (weight_vals - zero_val) * group_scales[k - start_idx]
    
    # 矩阵乘法
    result = torch.matmul(input, dequantized_weight.T)
    return result

def test_gptq_format():
    """测试GPTQ格式是否正确"""
    print("🧪 测试GPTQ格式修复...")
    
    # 使用较小的矩阵尺寸来加速测试
    M, K, N = 8, 512, 64  # 输入维度512，输出维度64
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
        
        # 测试基线实现（使用更快的版本）
        print("🐌 测试基线实现...")
        baseline_result = fast_baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 基线实现成功！输出形状: {baseline_result.shape}")
        
        # 可选：也测试原始基线实现（更慢但更准确）
        print("🐌 测试原始基线实现（较慢）...")
        original_baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 原始基线实现成功！输出形状: {original_baseline_result.shape}")
        
        # 比较结果
        diff1 = torch.abs(result - baseline_result).max()
        diff2 = torch.abs(result - original_baseline_result).max()
        diff3 = torch.abs(baseline_result - original_baseline_result).max()
        
        print(f"📊 融合算子 vs 快速基线最大差异: {diff1.item():.6f}")
        print(f"📊 融合算子 vs 原始基线最大差异: {diff2.item():.6f}")
        print(f"📊 快速基线 vs 原始基线最大差异: {diff3.item():.6f}")
        
        if diff1 < 1e-3 and diff2 < 1e-3:
            print("🎉 测试通过！所有实现结果一致")
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
