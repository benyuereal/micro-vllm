#!/usr/bin/env python3
"""
CUDA融合内核功能测试
测试gptq.py的功能和性能
"""

import torch
import sys
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from core.layer.gptq import GPTQCUDAFusion

def test_gptq_functionality():
    """测试GPTQ功能"""
    print("🚀 CUDA融合内核功能测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用，无法测试")
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 创建GPTQ融合实例
    print("\n🔨 创建GPTQ CUDA融合实例...")
    try:
        gptq_fusion = GPTQCUDAFusion(groupsize=128)
        print("✅ GPTQ CUDA融合实例创建成功")
    except Exception as e:
        raise RuntimeError(f"❌ GPTQ CUDA融合实例创建失败: {e}")
    
    # 测试用例 - 基于实际layer层数据形状
    # 注意：attn_output是[batch_size, num_heads, head_size] = [1, 32, 128]
    # 需要reshape为[batch_size*num_heads, head_size] = [32, 128]进行投影
    test_cases = [
        {
            "name": "QKV投影",
            "description": "input[1,4096] -> output[1,12288] (3*hidden_size)",
            "M": 1, "K": 4096, "N": 12288, "groupsize": 128,
            "input_shape": [1, 4096], "output_shape": [1, 12288]
        },
        {
            "name": "注意力输出投影", 
            "description": "input[32,128] -> output[32,128] (num_heads*head_size)",
            "M": 32, "K": 128, "N": 128, "groupsize": 128,
            "input_shape": [32, 128], "output_shape": [32, 128]
        },
        {
            "name": "输出投影",
            "description": "input[1,4096] -> output[1,4096] (hidden_size)",
            "M": 1, "K": 4096, "N": 4096, "groupsize": 128,
            "input_shape": [1, 4096], "output_shape": [1, 4096]
        },
        {
            "name": "MLP投影1",
            "description": "input[1,4096] -> output[1,11008] (intermediate_size)", 
            "M": 1, "K": 4096, "N": 11008, "groupsize": 128,
            "input_shape": [1, 4096], "output_shape": [1, 11008]
        },
        {
            "name": "MLP投影2",
            "description": "input[1,11008] -> output[1,4096] (hidden_size)",
            "M": 1, "K": 11008, "N": 4096, "groupsize": 128,
            "input_shape": [1, 11008], "output_shape": [1, 4096]
        }
    ]
    
    all_results = []
    
    for test_case in test_cases:
        print(f"\n⚡ 测试 {test_case['name']}")
        print("-" * 40)
        print(f"📊 {test_case['description']}")
        
        M, K, N = test_case["M"], test_case["K"], test_case["N"]
        groupsize = test_case["groupsize"]
        num_groups = K // groupsize
        
        # 创建测试数据
        input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        print(f"📊 输入形状: {input_tensor.shape} (期望: {test_case['input_shape']})")
        print(f"📊 qweight形状: {qweight.shape}")
        print(f"📊 qzeros形状: {qzeros.shape}")
        print(f"📊 scales形状: {scales.shape}")
        
        # 功能测试
        print("🧪 功能测试...")
        try:
            output = gptq_fusion.fused_gptq_gemm_4bit(
                input_tensor, qweight, qzeros, scales
            )
            print("✅ 功能测试成功!")
            print(f"📊 输出形状: {output.shape} (期望: {test_case['output_shape']})")
            print(f"📊 输出数据类型: {output.dtype}")
            
            # 验证输出
            expected_shape = tuple(test_case['output_shape'])
            assert output.shape == expected_shape, f"输出形状错误: 期望{expected_shape}, 实际{output.shape}"
            assert output.dtype == torch.float16, f"输出数据类型错误: 期望torch.float16, 实际{output.dtype}"
            assert not torch.isnan(output).any(), "输出包含NaN值"
            assert not torch.isinf(output).any(), "输出包含Inf值"
            
            print("✅ 输出验证通过!")
            
        except Exception as e:
            print(f"❌ 功能测试失败: {e}")
            raise RuntimeError(f"{test_case['name']} 功能测试失败: {e}")
        
        # 性能测试
        print("⚡ 性能测试...")
        try:
            result = gptq_fusion.benchmark(
                input_tensor, qweight, qzeros, scales, num_runs=50
            )
            
            print(f"📊 {test_case['name']} 性能统计:")
            print(f"  平均时间: {result['avg_time']:.2f}ms")
            print(f"  最小时间: {result['min_time']:.2f}ms")
            print(f"  最大时间: {result['max_time']:.2f}ms")
            print(f"  目标达成: {'✅' if result['target_achieved'] else '❌'} (目标: 0.10ms)")
            
            all_results.append({
                "name": test_case['name'],
                "avg_time": result['avg_time'],
                "target_achieved": result['target_achieved'],
                "description": test_case['description']
            })
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            raise RuntimeError(f"{test_case['name']} 性能测试失败: {e}")
    
    # 总体评估
    print(f"\n📊 总体功能测试结果")
    print("=" * 50)
    
    achieved_count = sum(1 for r in all_results if r['target_achieved'])
    total_count = len(all_results)
    
    print(f"功能测试通过率: {total_count}/{total_count} (100%)")
    print(f"性能目标达成率: {achieved_count}/{total_count} ({achieved_count/total_count*100:.1f}%)")
    
    print(f"\n📊 详细结果:")
    for result in all_results:
        status = "✅" if result['target_achieved'] else "❌"
        print(f"  {status} {result['name']}: {result['avg_time']:.2f}ms - {result['description']}")
    
    if achieved_count == total_count:
        print("🎉 所有测试用例都达到目标性能!")
    elif achieved_count > 0:
        print("✅ 部分测试用例达到目标性能")
    else:
        print("❌ 所有测试用例都未达到目标性能，需要进一步优化")
    
    print("\n🎉 CUDA融合内核功能测试完成!")
    return all_results

if __name__ == "__main__":
    try:
        results = test_gptq_functionality()
        
        # 输出测试结果摘要
        print(f"\n📋 测试结果摘要:")
        for result in results:
            print(f"  {result['name']}: {result['avg_time']:.2f}ms")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
