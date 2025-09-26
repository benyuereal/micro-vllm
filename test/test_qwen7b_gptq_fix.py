#!/usr/bin/env python3
"""
测试Qwen7B GPTQ格式修复
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

def test_qwen7b_gptq_format():
    """测试Qwen7B的GPTQ格式修复"""
    print("🧪 测试Qwen7B GPTQ格式修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟实际的Qwen7B GPTQ参数
    # 从日志: qweight=[512, 12288], qzeros=[32, 1536], scales=[32, 12288]
    N_compressed = 512  # 压缩的N维度
    K = 12288  # 输出维度 (QKV)
    input_K = 4096  # 输入维度
    num_groups = 32
    groupsize = 128
    
    print(f"Qwen7B GPTQ参数:")
    print(f"  qweight: [{N_compressed}, {K}]")
    print(f"  qzeros: [{num_groups}, {K//8}]")
    print(f"  scales: [{num_groups}, {K}]")
    print(f"  输入维度: {input_K}")
    print(f"  期望输出维度: {K}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_K, dtype=torch.bfloat16, device='cuda')
    
    # 模拟实际的GPTQ参数
    qweight = torch.randint(0, 256, (N_compressed, K), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"\n测试参数:")
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
        output_dim = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim}")
        
        # 验证输出维度
        if output_dim == K:
            print(f"🎉 输出维度正确: {output_dim}")
            
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
                print(f"❌ 输出维度 {output_dim} 不能被3整除")
                return False
        else:
            print(f"❌ 输出维度错误: 期望 {K}, 实际 {output_dim}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen7b_attention_structure():
    """测试Qwen7B的注意力结构"""
    print("\n🧪 测试Qwen7B注意力结构...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # Qwen7B注意力参数
    hidden_size = 4096
    num_heads = 32
    head_size = hidden_size // num_heads  # 128
    qkv_output_dim = hidden_size * 3  # 12288
    
    print(f"Qwen7B注意力结构:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_size: {head_size}")
    print(f"  QKV输出维度: {qkv_output_dim}")
    print(f"  分割后: Q({hidden_size}) + K({hidden_size}) + V({hidden_size}) = {qkv_output_dim}")
    
    # 模拟QKV投影后的分割
    qkv_output = torch.randn(1, qkv_output_dim, dtype=torch.bfloat16, device='cuda')
    print(f"  QKV输出形状: {qkv_output.shape}")
    
    # 分割QKV
    q, k, v = qkv_output.split(hidden_size, dim=-1)
    print(f"  分割后: Q{q.shape}, K{k.shape}, V{v.shape}")
    
    # 重塑为注意力格式
    batch_size, seq_len = 1, 1
    q = q.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
    k = k.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
    v = v.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
    
    print(f"  重塑后: Q{q.shape}, K{k.shape}, V{v.shape}")
    print(f"  注意力计算形状: Q{q.shape} @ K{k.shape}.transpose(-2, -1)")
    
    # 验证形状
    expected_shape = (batch_size, num_heads, seq_len, head_size)
    if q.shape == expected_shape and k.shape == expected_shape and v.shape == expected_shape:
        print(f"🎉 注意力形状正确: {expected_shape}")
        return True
    else:
        print(f"❌ 注意力形状错误: 期望 {expected_shape}")
        return False

def main():
    """主测试函数"""
    print("🚀 Qwen7B GPTQ格式修复测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("Qwen7B GPTQ格式修复测试", test_qwen7b_gptq_format),
        ("Qwen7B注意力结构测试", test_qwen7b_attention_structure)
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
