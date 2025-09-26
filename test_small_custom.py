#!/usr/bin/env python3
"""
测试小矩阵的自定义GPTQ格式
"""
import torch
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def test_small_custom_format():
    """测试小矩阵的自定义GPTQ格式"""
    print("🧪 测试小矩阵自定义GPTQ格式...")
    
    # 使用很小的矩阵尺寸
    M, K, N = 4, 128, 32  # 非常小的尺寸
    groupsize = 128
    
    # 创建自定义格式的张量
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 自定义格式：qweight [N, N//8]
    qweight = torch.randint(0, 256, (N, N // 8), dtype=torch.int32, device='cuda')  # [32, 4]
    
    # 自定义格式：scales [num_groups, K]
    num_groups = K // groupsize
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')  # [1, 128]
    
    # 自定义格式：qzeros [num_groups, custom_cols] - 使用较小的自定义列数
    qzeros = torch.randint(0, 16, (num_groups, 16), dtype=torch.int32, device='cuda')  # [1, 16]
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"qweight形状: {qweight.shape} (自定义格式: [N, N//8])")
    print(f"scales形状: {scales.shape} (标准格式: [num_groups, K])")
    print(f"qzeros形状: {qzeros.shape} (自定义格式: [num_groups, 16])")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试融合算子（应该自动回退到基线实现）
        print("🚀 测试融合算子...")
        start_time = time.time()
        result = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        fusion_time = time.time() - start_time
        print(f"✅ 融合算子成功！输出形状: {result.shape}, 耗时: {fusion_time:.4f}s")
        
        # 测试基线实现
        print("🐌 测试基线实现...")
        start_time = time.time()
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        baseline_time = time.time() - start_time
        print(f"✅ 基线实现成功！输出形状: {baseline_result.shape}, 耗时: {baseline_time:.4f}s")
        
        # 比较结果
        diff = torch.abs(result - baseline_result).max()
        speedup = baseline_time / fusion_time if fusion_time > 0 else float('inf')
        
        print(f"📊 最大差异: {diff.item():.6f}")
        print(f"🚀 加速比: {speedup:.2f}x")
        
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
    
    test_small_custom_format()
