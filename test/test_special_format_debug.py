#!/usr/bin/env python3
"""
测试特殊GPTQ格式修复 - 模拟实际推理场景
"""
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer.gptq import GPTQTritonFusion

def test_special_format_debug():
    """测试特殊格式并输出调试信息"""
    print("🧪 测试特殊GPTQ格式调试...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    # 模拟实际推理场景
    M, K, N = 1, 4096, 12288  # batch_size=1, input_dim=4096, output_dim=12288
    groupsize = 128
    
    # 创建测试数据 - 模拟实际的GPTQ格式
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 模拟实际的GPTQ参数格式
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    
    # 特殊格式：scales的第二维是N而不是K
    scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda')  # [num_groups, N]
    
    print(f"测试矩阵大小: {M}x{K}x{N}")
    print(f"qweight形状: {qweight.shape}")
    print(f"qzeros形状: {qzeros.shape}")
    print(f"scales形状: {scales.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试基线实现
        print("\n🐌 测试基线实现...")
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 基线实现成功: 输出形状 {baseline_result.shape}")
        
        # 验证输出形状
        expected_shape = (M, N)
        if baseline_result.shape == expected_shape:
            print(f"🎉 输出形状正确: {baseline_result.shape}")
        else:
            print(f"❌ 输出形状错误: 期望 {expected_shape}, 实际 {baseline_result.shape}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_special_format_debug()
