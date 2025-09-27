#!/usr/bin/env python3
"""
vLLM融合内核集成测试
测试vLLM内核与现有系统的集成
"""

import torch
import sys
import os
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_vllm_integration():
    """测试vLLM内核集成"""
    print("🚀 vLLM融合内核集成测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用，无法测试")
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 编译vLLM内核
    print("\n📦 编译vLLM内核...")
    from torch.utils.cpp_extension import load
    
    # 切换到cuda目录
    cuda_dir = os.path.join(project_root, "cuda")
    original_cwd = os.getcwd()
    os.chdir(cuda_dir)
    
    try:
        kernel_module = load(
            name="fused_gptq_gemm_4bit_cuda",
            sources=["gptq_cuda_kernel_vllm.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        print("✅ vLLM内核编译成功")
    except Exception as e:
        print(f"❌ vLLM内核编译失败: {e}")
        return False
    finally:
        os.chdir(original_cwd)
    
    # 测试数据
    print("\n📊 测试数据准备...")
    batch_size, seq_len, hidden_dim = 1, 1, 4096
    groupsize = 128
    num_groups = hidden_dim // groupsize
    
    # 输入数据
    input_tensor = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.float16, device='cuda')
    
    # GPTQ参数
    qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
    qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
    scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
    
    print(f"📊 输入形状: {input_tensor.shape}")
    print(f"📊 qweight形状: {qweight.shape}")
    print(f"📊 qzeros形状: {qzeros.shape}")
    print(f"📊 scales形状: {scales.shape}")
    
    # 测试vLLM内核
    print("\n⚡ 测试vLLM内核...")
    try:
        # 预热
        for _ in range(5):
            output = kernel_module.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        
        # 功能测试
        output = kernel_module.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 期望形状: torch.Size([1, 12288])")
        
        expected_shape = (batch_size * seq_len, hidden_dim * 3)
        if output.shape != expected_shape:
            print(f"❌ 输出形状错误: {output.shape} != {expected_shape}")
            return False
        
        print("✅ vLLM内核功能测试通过!")
        
        # 性能测试
        print("\n⏱️ 性能测试...")
        times = []
        for i in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = kernel_module.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"📊 性能统计:")
        print(f"  平均时间: {avg_time:.3f}ms")
        print(f"  最小时间: {min_time:.3f}ms")
        print(f"  最大时间: {max_time:.3f}ms")
        
        # 检查是否达到目标
        if avg_time <= 0.3:
            print(f"🎯 达到性能目标: {avg_time:.3f}ms <= 0.3ms")
        else:
            print(f"⚠️ 未达到性能目标: {avg_time:.3f}ms > 0.3ms")
        
        return True
        
    except Exception as e:
        print(f"❌ vLLM内核测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gptq_cuda_fusion_integration():
    """测试GPTQCUDAFusion集成"""
    print("\n🔨 测试GPTQCUDAFusion集成...")
    
    try:
        from core.layer.gptq import GPTQCUDAFusion
        
        # 创建GPTQ融合实例
        gptq_fusion = GPTQCUDAFusion(groupsize=128)
        print("✅ GPTQCUDAFusion实例创建成功")
        
        # 测试数据
        M, K, N = 1, 4096, 12288
        input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (K // 128, K // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(K // 128, K, dtype=torch.float16, device='cuda')
        
        # 执行测试
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input_tensor, qweight, qzeros, scales
        )
        
        print(f"📊 GPTQCUDAFusion输出形状: {output.shape}")
        print(f"📊 期望形状: torch.Size([1, 12288])")
        
        if output.shape == (1, 12288):
            print("✅ GPTQCUDAFusion集成测试通过!")
            return True
        else:
            print(f"❌ GPTQCUDAFusion输出形状错误: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ GPTQCUDAFusion集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始vLLM融合内核集成测试...")
    
    success = True
    
    # 测试vLLM内核集成
    if not test_vllm_integration():
        success = False
    
    # 测试GPTQCUDAFusion集成
    if not test_gptq_cuda_fusion_integration():
        success = False
    
    if success:
        print("\n🎉 所有集成测试通过!")
    else:
        print("\n💥 部分集成测试失败!")
    
    return success

if __name__ == "__main__":
    main()
