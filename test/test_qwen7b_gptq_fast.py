#!/usr/bin/env python3
"""
快速测试Qwen7B GPTQ格式修复
"""
import torch
import sys
import os
import logging
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO)

from core.layer.gptq import GPTQTritonFusion

def test_qwen7b_gptq_format_fast():
    """快速测试Qwen7B的GPTQ格式修复"""
    print("🧪 快速测试Qwen7B GPTQ格式修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 使用更小的矩阵进行快速测试
    input_dim = 64  # 减小输入维度
    output_dim = 192  # 减小输出维度 (3 * 64)
    groupsize = 32
    num_groups = output_dim // groupsize
    
    print(f"快速测试参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟Qwen7B的GPTQ格式: [input_dim//8, output_dim]
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
        start_time = time.time()
        
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        
        end_time = time.time()
        print(f"✅ 基线实现成功: 输出形状 {baseline_result.shape}")
        print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
        
        # 检查输出维度
        output_dim_actual = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim_actual}")
        
        # 验证输出维度
        if output_dim_actual == output_dim:
            print(f"🎉 输出维度正确: {output_dim_actual}")
            
            # 测试分割
            if output_dim % 3 == 0:
                hidden_size = output_dim // 3
                print(f"✅ 可以分割为3部分，每部分维度: {hidden_size}")
                
                # 测试分割
                q, k, v = baseline_result.split(hidden_size, dim=-1)
                print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
                
                # 验证分割后的维度
                if q.shape[-1] == hidden_size and k.shape[-1] == hidden_size and v.shape[-1] == hidden_size:
                    print(f"🎉 分割维度正确: 每个部分都是 {hidden_size} 维度")
                    return True
                else:
                    print(f"❌ 分割维度错误")
                    return False
            else:
                print(f"❌ 输出维度 {output_dim_actual} 不能被3整除")
                return False
        else:
            print(f"❌ 输出维度错误: 期望 {output_dim}, 实际 {output_dim_actual}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen7b_gptq_format_full():
    """完整测试Qwen7B的GPTQ格式修复"""
    print("\n🧪 完整测试Qwen7B GPTQ格式修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 使用完整的Qwen7B参数
    input_dim = 4096
    output_dim = 12288
    groupsize = 128
    num_groups = output_dim // groupsize
    
    print(f"完整测试参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟Qwen7B的GPTQ格式: [input_dim//8, output_dim]
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
        start_time = time.time()
        
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        
        end_time = time.time()
        print(f"✅ 基线实现成功: 输出形状 {baseline_result.shape}")
        print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
        
        # 检查输出维度
        output_dim_actual = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim_actual}")
        
        # 验证输出维度
        if output_dim_actual == output_dim:
            print(f"🎉 输出维度正确: {output_dim_actual}")
            
            # 测试分割
            if output_dim % 3 == 0:
                hidden_size = output_dim // 3
                print(f"✅ 可以分割为3部分，每部分维度: {hidden_size}")
                
                # 测试分割
                q, k, v = baseline_result.split(hidden_size, dim=-1)
                print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
                
                # 验证分割后的维度
                if q.shape[-1] == hidden_size and k.shape[-1] == hidden_size and v.shape[-1] == hidden_size:
                    print(f"🎉 分割维度正确: 每个部分都是 {hidden_size} 维度")
                    return True
                else:
                    print(f"❌ 分割维度错误")
                    return False
            else:
                print(f"❌ 输出维度 {output_dim_actual} 不能被3整除")
                return False
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
    print("🚀 Qwen7B GPTQ格式修复快速测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("快速测试", test_qwen7b_gptq_format_fast),
        ("完整测试", test_qwen7b_gptq_format_full)
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
        print("🎉 所有Qwen7B GPTQ格式修复测试通过！")
    else:
        print("⚠️  部分Qwen7B GPTQ格式修复测试失败")

if __name__ == "__main__":
    main()
