#!/usr/bin/env python3
"""
测试数据类型修复 - BFloat16和Float16兼容性
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

def test_dtype_compatibility():
    """测试数据类型兼容性"""
    print("🧪 测试数据类型兼容性...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 测试不同的数据类型
    dtypes = [torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        print(f"\n🔍 测试数据类型: {dtype}")
        
        # 模拟实际推理场景 - 混合格式
        M, K, N = 1, 4096, 512
        groupsize = 128
        
        # 创建测试数据 - 使用指定的数据类型
        input_tensor = torch.randn(M, K, dtype=dtype, device='cuda')
        
        # 模拟实际的GPTQ参数格式（混合格式）
        num_groups = K // groupsize
        qweight = torch.randint(0, 256, (N, N), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda')
        
        print(f"输入数据类型: {input_tensor.dtype}")
        print(f"scales数据类型: {scales.dtype}")
        
        # 创建GPTQ融合对象
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
            
            print(f"✅ 基线实现成功: 输出形状 {baseline_result.shape}, 数据类型 {baseline_result.dtype}")
            
            # 验证输出数据类型
            if baseline_result.dtype == input_tensor.dtype:
                print(f"🎉 数据类型匹配: {baseline_result.dtype}")
            else:
                print(f"❌ 数据类型不匹配: 期望 {input_tensor.dtype}, 实际 {baseline_result.dtype}")
                return False
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_mixed_dtype_scenarios():
    """测试混合数据类型场景"""
    print("\n🧪 测试混合数据类型场景...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 测试场景：输入是BFloat16，scales是Float16
    M, K, N = 1, 4096, 512
    groupsize = 128
    
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (N, N), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, N, dtype=torch.float16, device='cuda')
    
    print(f"输入数据类型: {input_tensor.dtype}")
    print(f"scales数据类型: {scales.dtype}")
    
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        
        print(f"✅ 混合数据类型测试成功: 输出形状 {baseline_result.shape}, 数据类型 {baseline_result.dtype}")
        
        # 验证输出数据类型
        if baseline_result.dtype == input_tensor.dtype:
            print(f"🎉 数据类型正确匹配: {baseline_result.dtype}")
            return True
        else:
            print(f"❌ 数据类型不匹配: 期望 {input_tensor.dtype}, 实际 {baseline_result.dtype}")
            return False
            
    except Exception as e:
        print(f"❌ 混合数据类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 数据类型兼容性测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("数据类型兼容性测试", test_dtype_compatibility),
        ("混合数据类型场景测试", test_mixed_dtype_scenarios)
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
        print("🎉 所有数据类型测试通过！")
    else:
        print("⚠️  部分数据类型测试失败")

if __name__ == "__main__":
    main()
