#!/usr/bin/env python3
"""
测试输出投影的GPTQ格式
"""
import torch
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO)

from core.layer.gptq import GPTQTritonFusion

def test_output_projection_gptq_format():
    """测试输出投影的GPTQ格式"""
    print("🧪 测试输出投影的GPTQ格式...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟输出投影的参数
    # 输出投影: 输入128 → 输出4096
    input_dim = 128  # 注意力输出维度
    output_dim = 4096  # 隐藏维度
    groupsize = 128
    num_groups = output_dim // groupsize  # 32
    
    print(f"输出投影参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟输出投影的GPTQ格式
    # 这应该是 [output_dim, input_dim//8] 格式
    qweight = torch.randint(0, 256, (output_dim, input_dim // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, input_dim // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, input_dim, dtype=torch.float16, device='cuda')
    
    print(f"测试参数:")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  qweight形状: {qweight.shape}")
    print(f"  qzeros形状: {qzeros.shape}")
    print(f"  scales形状: {scales.shape}")
    
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
        
        # 检查输出维度
        output_dim_actual = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim_actual}")
        
        # 验证输出维度
        if output_dim_actual == output_dim:
            print(f"🎉 输出维度正确: {output_dim_actual}")
            return True
        else:
            print(f"❌ 输出维度错误: 期望 {output_dim}, 实际 {output_dim_actual}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_projection_gptq_format_alternative():
    """测试输出投影的GPTQ格式 - 替代格式"""
    print("\n🧪 测试输出投影的GPTQ格式 - 替代格式...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟输出投影的参数
    # 输出投影: 输入128 → 输出4096
    input_dim = 128  # 注意力输出维度
    output_dim = 4096  # 隐藏维度
    groupsize = 128
    num_groups = output_dim // groupsize  # 32
    
    print(f"输出投影参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟输出投影的GPTQ格式 - 使用Qwen7B格式
    # 这可能是 [input_dim//8, output_dim] 格式
    qweight = torch.randint(0, 256, (input_dim // 8, output_dim), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, output_dim // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, output_dim, dtype=torch.float16, device='cuda')
    
    print(f"测试参数:")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  qweight形状: {qweight.shape}")
    print(f"  qzeros形状: {qzeros.shape}")
    print(f"  scales形状: {scales.shape}")
    
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
        
        # 检查输出维度
        output_dim_actual = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim_actual}")
        
        # 验证输出维度
        if output_dim_actual == output_dim:
            print(f"🎉 输出维度正确: {output_dim_actual}")
            return True
        else:
            print(f"❌ 输出维度错误: 期望 {output_dim}, 实际 {output_dim_actual}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 输出投影GPTQ格式测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("标准格式测试", test_output_projection_gptq_format),
        ("Qwen7B格式测试", test_output_projection_gptq_format_alternative)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有输出投影GPTQ格式测试通过！")
    else:
        print("⚠️  部分输出投影GPTQ格式测试失败")

if __name__ == "__main__":
    main()
