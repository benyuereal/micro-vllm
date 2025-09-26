#!/usr/bin/env python3
"""
测试QKV投影输出维度修复
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

def test_qkv_output_dimension():
    """测试QKV投影输出维度"""
    print("🧪 测试QKV投影输出维度...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟实际推理场景
    M, K, N = 1, 4096, 512  # batch_size=1, input_dim=4096, output_dim=512
    groupsize = 128
    
    # 创建测试数据
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    # 模拟实际的GPTQ参数格式
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, N), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda')
    
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
        
        # 检查输出维度
        output_dim = baseline_result.shape[-1]
        print(f"📊 输出维度: {output_dim}")
        
        # 检查是否可以分割为3部分
        if output_dim % 3 == 0:
            hidden_size = output_dim // 3
            print(f"✅ 可以分割为3部分，每部分维度: {hidden_size}")
            
            # 测试分割
            q, k, v = baseline_result.split(hidden_size, dim=-1)
            print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
            return True
        else:
            print(f"❌ 输出维度 {output_dim} 不能被3整除")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expected_qkv_dimensions():
    """测试期望的QKV维度"""
    print("\n🧪 测试期望的QKV维度...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟Qwen7B的QKV投影
    # 通常QKV投影的输出维度是 3 * hidden_size
    hidden_size = 4096  # Qwen7B的hidden_size
    expected_qkv_dim = hidden_size * 3  # 12288
    
    print(f"期望的QKV输出维度: {expected_qkv_dim}")
    
    # 创建测试数据
    M, K = 1, hidden_size
    groupsize = 128
    
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    # 创建正确维度的GPTQ参数
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (expected_qkv_dim, expected_qkv_dim), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, expected_qkv_dim, dtype=torch.float16, device='cuda')
    
    print(f"测试矩阵大小: {M}x{K}x{expected_qkv_dim}")
    print(f"qweight形状: {qweight.shape}")
    print(f"scales形状: {scales.shape}")
    
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        
        print(f"✅ 结果形状: {result.shape}")
        
        # 检查输出维度
        output_dim = result.shape[-1]
        if output_dim == expected_qkv_dim:
            print(f"🎉 输出维度正确: {output_dim}")
            
            # 测试分割
            hidden_size = output_dim // 3
            q, k, v = result.split(hidden_size, dim=-1)
            print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
            return True
        else:
            print(f"❌ 输出维度错误: 期望 {expected_qkv_dim}, 实际 {output_dim}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 QKV投影输出维度测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("QKV投影输出维度测试", test_qkv_output_dimension),
        ("期望QKV维度测试", test_expected_qkv_dimensions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 测试总结")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有QKV投影测试通过！")
    else:
        print("⚠️  部分QKV投影测试失败")

if __name__ == "__main__":
    main()
