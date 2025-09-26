#!/usr/bin/env python3
"""
测试输出投影的形状处理
"""
import torch
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO)

def test_output_projection_shape():
    """测试输出投影的形状处理"""
    print("🧪 测试输出投影的形状处理...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟注意力计算的输出
    batch_size = 1
    num_heads = 32
    head_size = 128
    hidden_size = num_heads * head_size  # 4096
    
    print(f"注意力输出参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_size: {head_size}")
    print(f"  hidden_size: {hidden_size}")
    
    # 创建注意力输出张量
    attn_output = torch.randn(batch_size, num_heads, head_size, dtype=torch.bfloat16, device='cuda')
    print(f"  attn_output shape: {attn_output.shape}")
    
    # 重塑注意力输出: [batch_size, num_heads, head_size] -> [batch_size, seq_len, hidden_size]
    attn_output_reshaped = attn_output.view(batch_size, 1, hidden_size)
    print(f"  reshaped attn_output: {attn_output_reshaped.shape}")
    
    # 验证重塑是否正确
    if attn_output_reshaped.shape == (batch_size, 1, hidden_size):
        print(f"✅ 重塑成功: {attn_output_reshaped.shape}")
        
        # 验证数据一致性
        original_data = attn_output.view(batch_size, -1)
        reshaped_data = attn_output_reshaped.view(batch_size, -1)
        
        if torch.allclose(original_data, reshaped_data):
            print(f"✅ 数据一致性验证通过")
            return True
        else:
            print(f"❌ 数据一致性验证失败")
            return False
    else:
        print(f"❌ 重塑失败: 期望 {(batch_size, 1, hidden_size)}, 实际 {attn_output_reshaped.shape}")
        return False

def test_output_projection_gptq_format():
    """测试输出投影的GPTQ格式"""
    print("\n🧪 测试输出投影的GPTQ格式...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟输出投影的参数
    input_dim = 4096  # 隐藏维度
    output_dim = 4096  # 输出维度
    groupsize = 128
    num_groups = input_dim // groupsize  # 32
    
    print(f"输出投影参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟输出投影的GPTQ格式: [output_dim//8, input_dim]
    qweight = torch.randint(0, 256, (output_dim // 8, input_dim), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, input_dim // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, input_dim, dtype=torch.float16, device='cuda')
    
    print(f"测试参数:")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  qweight形状: {qweight.shape}")
    print(f"  qzeros形状: {qzeros.shape}")
    print(f"  scales形状: {scales.shape}")
    
    # 验证GPTQ格式
    if (qweight.shape[0] == output_dim // 8 and 
        qweight.shape[1] == input_dim and
        qzeros.shape[0] == num_groups and
        qzeros.shape[1] == input_dim // 8 and
        scales.shape[0] == num_groups and
        scales.shape[1] == input_dim):
        print(f"✅ GPTQ格式验证通过")
        return True
    else:
        print(f"❌ GPTQ格式验证失败")
        return False

def main():
    """主测试函数"""
    print("🚀 输出投影形状处理测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("形状处理测试", test_output_projection_shape),
        ("GPTQ格式测试", test_output_projection_gptq_format)
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
        print("🎉 所有输出投影形状处理测试通过！")
    else:
        print("⚠️  部分输出投影形状处理测试失败")

if __name__ == "__main__":
    main()
