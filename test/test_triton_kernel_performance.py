#!/usr/bin/env python3
"""
测试Triton融合内核性能
"""
import torch
import time
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.layer.gptq import GPTQTritonFusion

def test_triton_kernel_qwen7b_qkv():
    """测试Qwen7B QKV投影的Triton内核性能"""
    print("🧪 测试Qwen7B QKV投影Triton内核性能...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # Qwen7B QKV投影参数
    input_dim = 4096  # 隐藏维度
    output_dim = 12288  # QKV输出维度 (3 * 4096)
    groupsize = 128
    num_groups = input_dim // groupsize  # 32
    
    print(f"Qwen7B QKV投影参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟Qwen7B QKV投影的GPTQ格式: [input_dim//8, output_dim]
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
        # 预热
        print("\n🔥 预热Triton内核...")
        for _ in range(3):
            _ = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
        torch.cuda.synchronize()
        
        # 性能测试
        print("\n⚡ 性能测试...")
        num_iterations = 10
        times = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            
            result = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000  # 转换为毫秒
            times.append(elapsed)
            
            if i < 3:  # 前几次可能较慢，跳过
                continue
            print(f"  迭代 {i+1}: {elapsed:.2f}ms")
        
        # 计算统计信息
        valid_times = times[3:]  # 跳过前3次
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        
        print(f"\n📊 Qwen7B QKV投影性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        print(f"  结果形状: {result.shape}")
        
        # 验证结果
        expected_shape = (1, output_dim)
        if result.shape == expected_shape:
            print(f"✅ 结果形状正确: {result.shape}")
            return True
        else:
            print(f"❌ 结果形状错误: 期望 {expected_shape}, 实际 {result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triton_kernel_qwen7b_output():
    """测试Qwen7B输出投影的Triton内核性能"""
    print("\n🧪 测试Qwen7B输出投影Triton内核性能...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # Qwen7B输出投影参数
    input_dim = 4096  # 隐藏维度
    output_dim = 4096  # 输出维度
    groupsize = 128
    num_groups = input_dim // groupsize  # 32
    
    print(f"Qwen7B输出投影参数:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    print(f"  分组大小: {groupsize}")
    print(f"  分组数量: {num_groups}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    
    # 模拟Qwen7B输出投影的GPTQ格式: [output_dim//8, input_dim]
    qweight = torch.randint(0, 256, (output_dim // 8, input_dim), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, output_dim // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, input_dim, dtype=torch.float16, device='cuda')
    
    print(f"测试参数:")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  qweight形状: {qweight.shape}")
    print(f"  qzeros形状: {qzeros.shape}")
    print(f"  scales形状: {scales.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 预热
        print("\n🔥 预热Triton内核...")
        for _ in range(3):
            _ = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
        torch.cuda.synchronize()
        
        # 性能测试
        print("\n⚡ 性能测试...")
        num_iterations = 10
        times = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            
            result = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000  # 转换为毫秒
            times.append(elapsed)
            
            if i < 3:  # 前几次可能较慢，跳过
                continue
            print(f"  迭代 {i+1}: {elapsed:.2f}ms")
        
        # 计算统计信息
        valid_times = times[3:]  # 跳过前3次
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        
        print(f"\n📊 Qwen7B输出投影性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        print(f"  结果形状: {result.shape}")
        
        # 验证结果
        expected_shape = (1, output_dim)
        if result.shape == expected_shape:
            print(f"✅ 结果形状正确: {result.shape}")
            return True
        else:
            print(f"❌ 结果形状错误: 期望 {expected_shape}, 实际 {result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """性能对比测试"""
    print("\n🏁 性能对比测试...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 测试参数
    input_dim = 4096
    output_dim = 12288  # QKV投影
    groupsize = 128
    num_groups = input_dim // groupsize
    
    # 创建测试数据
    input_tensor = torch.randn(1, input_dim, dtype=torch.bfloat16, device='cuda')
    qweight = torch.randint(0, 256, (input_dim // 8, output_dim), dtype=torch.int32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, output_dim // 8), dtype=torch.int32, device='cuda')
    scales = torch.randn(num_groups, output_dim, dtype=torch.float16, device='cuda')
    
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # Triton融合内核测试
        print("🔥 预热Triton内核...")
        for _ in range(3):
            _ = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
        torch.cuda.synchronize()
        
        print("⚡ Triton融合内核性能测试...")
        triton_times = []
        for i in range(10):
            torch.cuda.synchronize()
            start_time = time.time()
            result_triton = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
            torch.cuda.synchronize()
            elapsed = (time.time() - start_time) * 1000
            triton_times.append(elapsed)
        
        triton_avg = sum(triton_times[3:]) / len(triton_times[3:])
        
        print(f"\n📊 性能对比结果:")
        print(f"  Triton融合内核: {triton_avg:.2f}ms")
        print(f"  目标性能: 0.7ms")
        print(f"  性能提升: {triton_avg/0.7:.1f}x")
        
        if triton_avg < 1.0:  # 小于1ms认为性能良好
            print(f"✅ Triton内核性能优秀!")
            return True
        else:
            print(f"⚠️ Triton内核性能需要进一步优化")
            return False
            
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Triton融合内核性能测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("Qwen7B QKV投影", test_triton_kernel_qwen7b_qkv),
        ("Qwen7B输出投影", test_triton_kernel_qwen7b_output),
        ("性能对比", test_performance_comparison)
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
        print("🎉 所有Triton融合内核性能测试通过！")
    else:
        print("⚠️  部分Triton融合内核性能测试失败")

if __name__ == "__main__":
    main()
