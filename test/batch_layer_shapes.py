#!/usr/bin/env python3
"""
批处理Layer层测试
基于实际layer层数据形状的批处理测试
"""

import torch
import sys
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

def test_batch_layer_shapes():
    """测试批处理layer层形状"""
    print("🚀 批处理Layer层形状测试")
    print("=" * 60)
    
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
    
    # 基于实际layer层数据形状的测试用例
    print("\n📊 基于实际layer层数据形状的测试用例:")
    print("=" * 60)
    
    # Layer层参数
    batch_size = 1
    seq_len = 1
    hidden_size = 4096
    num_heads = 32
    head_size = 128
    intermediate_size = 11008
    
    print(f"📊 Layer层参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_size: {head_size}")
    print(f"  intermediate_size: {intermediate_size}")
    
    # 测试数据形状映射
    test_cases = [
        {
            "name": "QKV投影",
            "description": "hidden_states[1,1,4096] -> qkv[1,1,12288]",
            "input_shape": [batch_size, seq_len, hidden_size],
            "output_shape": [batch_size, seq_len, 3 * hidden_size],
            "gemm_shape": [batch_size * seq_len, hidden_size],  # 2D for GEMM
            "gemm_output": [batch_size * seq_len, 3 * hidden_size]
        },
        {
            "name": "注意力输出投影",
            "description": "attn_output[1,32,128] -> proj_fn[1,1,4096]",
            "input_shape": [batch_size, num_heads, head_size],
            "output_shape": [batch_size, seq_len, hidden_size],
            "gemm_shape": [batch_size * num_heads, head_size],  # 2D for GEMM
            "gemm_output": [batch_size * num_heads, head_size]
        },
        {
            "name": "最终输出投影",
            "description": "proj_fn[1,1,4096] -> proj[1,1,4096]",
            "input_shape": [batch_size, seq_len, hidden_size],
            "output_shape": [batch_size, seq_len, hidden_size],
            "gemm_shape": [batch_size * seq_len, hidden_size],  # 2D for GEMM
            "gemm_output": [batch_size * seq_len, hidden_size]
        },
        {
            "name": "MLP投影1",
            "description": "hidden_states[1,1,4096] -> mlp[1,1,11008]",
            "input_shape": [batch_size, seq_len, hidden_size],
            "output_shape": [batch_size, seq_len, intermediate_size],
            "gemm_shape": [batch_size * seq_len, hidden_size],  # 2D for GEMM
            "gemm_output": [batch_size * seq_len, intermediate_size]
        },
        {
            "name": "MLP投影2",
            "description": "mlp[1,1,11008] -> hidden_states[1,1,4096]",
            "input_shape": [batch_size, seq_len, intermediate_size],
            "output_shape": [batch_size, seq_len, hidden_size],
            "gemm_shape": [batch_size * seq_len, intermediate_size],  # 2D for GEMM
            "gemm_output": [batch_size * seq_len, hidden_size]
        }
    ]
    
    all_results = []
    
    for test_case in test_cases:
        print(f"\n⚡ 测试 {test_case['name']}")
        print("-" * 50)
        print(f"📊 {test_case['description']}")
        
        # 创建测试数据
        input_3d = torch.randn(*test_case['input_shape'], dtype=torch.float16, device='cuda')
        input_2d = input_3d.view(*test_case['gemm_shape'])
        
        M, K = test_case['gemm_shape']
        N = test_case['gemm_output'][1]
        groupsize = 128
        num_groups = K // groupsize
        
        # 创建量化权重
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        print(f"📊 3D输入形状: {input_3d.shape}")
        print(f"📊 2D输入形状: {input_2d.shape} (用于GEMM)")
        print(f"📊 qweight形状: {qweight.shape}")
        print(f"📊 期望2D输出: {test_case['gemm_output']}")
        print(f"📊 期望3D输出: {test_case['output_shape']}")
        
        # 功能测试
        print("🧪 功能测试...")
        try:
            output_2d = gptq_fusion.fused_gptq_gemm_4bit(
                input_2d, qweight, qzeros, scales
            )
            
            # 验证2D输出
            expected_2d_shape = tuple(test_case['gemm_output'])
            assert output_2d.shape == expected_2d_shape, f"2D输出形状错误: 期望{expected_2d_shape}, 实际{output_2d.shape}"
            
            # 验证3D输出
            output_3d = output_2d.view(*test_case['output_shape'])
            expected_3d_shape = tuple(test_case['output_shape'])
            assert output_3d.shape == expected_3d_shape, f"3D输出形状错误: 期望{expected_3d_shape}, 实际{output_3d.shape}"
            
            print("✅ 功能测试成功!")
            print(f"📊 2D输出形状: {output_2d.shape}")
            print(f"📊 3D输出形状: {output_3d.shape}")
            
        except Exception as e:
            print(f"❌ 功能测试失败: {e}")
            raise RuntimeError(f"{test_case['name']} 功能测试失败: {e}")
        
        # 性能测试
        print("⚡ 性能测试...")
        try:
            result = gptq_fusion.benchmark(
                input_2d, qweight, qzeros, scales, num_runs=50
            )
            
            print(f"📊 {test_case['name']} 性能统计:")
            print(f"  平均时间: {result['avg_time']:.2f}ms")
            print(f"  最小时间: {result['min_time']:.2f}ms")
            print(f"  最大时间: {result['max_time']:.2f}ms")
            print(f"  目标达成: {'✅' if result['target_achieved'] else '❌'} (目标: 0.20ms)")
            
            all_results.append({
                "name": test_case['name'],
                "avg_time": result['avg_time'],
                "target_achieved": result['target_achieved'],
                "description": test_case['description'],
                "gemm_shape": test_case['gemm_shape'],
                "output_shape": test_case['output_shape']
            })
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            raise RuntimeError(f"{test_case['name']} 性能测试失败: {e}")
    
    # 总体评估
    print(f"\n📊 批处理Layer层测试结果")
    print("=" * 60)
    
    achieved_count = sum(1 for r in all_results if r['target_achieved'])
    total_count = len(all_results)
    
    print(f"功能测试通过率: {total_count}/{total_count} (100%)")
    print(f"性能目标达成率: {achieved_count}/{total_count} ({achieved_count/total_count*100:.1f}%)")
    
    print(f"\n📊 详细结果:")
    for result in all_results:
        status = "✅" if result['target_achieved'] else "❌"
        print(f"  {status} {result['name']}: {result['avg_time']:.2f}ms")
        print(f"    GEMM形状: {result['gemm_shape']} -> {result['output_shape']}")
        print(f"    描述: {result['description']}")
    
    if achieved_count == total_count:
        print("🎉 所有测试用例都达到目标性能!")
    elif achieved_count > 0:
        print("✅ 部分测试用例达到目标性能")
    else:
        print("❌ 所有测试用例都未达到目标性能，需要进一步优化")
    
    print("\n🎉 批处理Layer层测试完成!")
    return all_results

if __name__ == "__main__":
    try:
        results = test_batch_layer_shapes()
        
        # 输出测试结果摘要
        print(f"\n📋 测试结果摘要:")
        for result in results:
            print(f"  {result['name']}: {result['avg_time']:.2f}ms")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
