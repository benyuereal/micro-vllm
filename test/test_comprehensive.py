#!/usr/bin/env python3
"""
GPTQ综合测试 - 包含所有重要测试功能
"""
import torch
import sys
import os
import time
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO)

from core.layer.gptq import GPTQTritonFusion

def test_mixed_format():
    """测试混合GPTQ格式（实际推理场景）"""
    print("🧪 测试混合GPTQ格式...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟实际推理场景 - 混合格式
    M, K, N = 1, 4096, 512  # batch_size=1, input_dim=4096, output_dim=512
    groupsize = 128
    
    # 创建测试数据 - 模拟混合的GPTQ格式
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 模拟实际的GPTQ参数格式（混合格式）
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, N), dtype=torch.int32, device='cuda')  # [N, N] - 混合格式
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')  # [num_groups, K//8]
    scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda')  # [num_groups, N] - 混合格式
    
    print(f"测试矩阵大小: {M}x{K}x{N}")
    print(f"qweight形状: {qweight.shape} (混合格式)")
    print(f"qzeros形状: {qzeros.shape}")
    print(f"scales形状: {scales.shape} (混合格式)")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试基线实现
        print("\n🐌 测试基线实现（混合格式）...")
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
            return True
        else:
            print(f"❌ 输出形状错误: 期望 {expected_shape}, 实际 {baseline_result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standard_format():
    """测试标准GPTQ格式"""
    print("\n🧪 测试标准GPTQ格式...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 标准格式测试
    M, K, N = 32, 4096, 12288
    groupsize = 128
    
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')  # 标准格式
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"标准格式测试: {M}x{K}x{N}")
    print(f"qweight形状: {qweight.shape} (标准格式)")
    
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 标准格式成功: 输出形状 {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 标准格式失败: {e}")
        return False

def test_performance():
    """测试性能优化效果"""
    print("\n🚀 测试性能优化效果...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 使用中等大小的矩阵
    M, K, N = 256, 1024, 512
    groupsize = 128
    
    # 创建测试数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    print(f"性能测试矩阵大小: {M}x{K}x{N}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    # 测试不同的反量化方法
    methods = {
        'vectorized': fusion.dequantize_gptq_weight,
        'batch_optimized': fusion.dequantize_gptq_weight_batch_optimized
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n🧪 测试 {method_name} 方法...")
        
        # 预热
        for _ in range(3):
            _ = method_func(qweight, qzeros, scales, groupsize)
        torch.cuda.synchronize()
        
        # 测试性能
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.time()
            dequantized_weight = method_func(qweight, qzeros, scales, groupsize)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
        results[method_name] = avg_time
        print(f"✅ {method_name}: {avg_time:.2f} ms")
    
    # 计算加速比
    print(f"\n📈 性能比较:")
    baseline_time = results['vectorized']
    for method_name, time_ms in results.items():
        speedup = baseline_time / time_ms
        print(f"{method_name}: {time_ms:.2f} ms (加速比: {speedup:.2f}x)")
    
    return True

def test_correctness():
    """测试正确性"""
    print("\n✅ 测试正确性...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 使用小矩阵进行正确性测试
    M, K, N = 32, 256, 128
    groupsize = 128
    
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
    
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试基线实现
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        
        # 测试融合算子
        fusion_result = fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales
        )
        
        # 比较结果
        diff = torch.abs(baseline_result - fusion_result).max()
        print(f"📊 最大差异: {diff.item():.6f}")
        
        if diff < 1e-3:
            print("🎉 结果一致！")
            return True
        else:
            print("⚠️  结果有差异")
            return False
            
    except Exception as e:
        print(f"❌ 正确性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 GPTQ综合测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("混合格式测试", test_mixed_format),
        ("标准格式测试", test_standard_format),
        ("性能测试", test_performance),
        ("正确性测试", test_correctness)
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
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")

if __name__ == "__main__":
    main()
