#!/usr/bin/env python3
"""
测试修复后的vLLM GPTQ内核
"""

import torch
from torch.utils.cpp_extension import load
import time
import numpy as np

def test_vllm_fixed():
    print("🚀 测试修复后的vLLM GPTQ内核...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    try:
        # 编译修复后的vLLM内核
        print("📦 编译修复后的vLLM内核...")
        vllm_kernel = load(
            name="fused_gptq_gemm_4bit_cuda",
            sources=["gptq_cuda_kernel_vllm.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=True
        )
        print("✅ vLLM内核编译成功!")
        
        # 测试参数
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        groupsize = 128
        num_groups = hidden_dim // groupsize
        
        # 创建测试数据
        print("🔧 创建测试数据...")
        input_tensor = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 255, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 255, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        print(f"📊 输入形状: {input_tensor.shape}")
        print(f"📊 qweight形状: {qweight.shape}")
        print(f"📊 qzeros形状: {qzeros.shape}")
        print(f"📊 scales形状: {scales.shape}")
        
        # 功能测试
        print("🧪 进行功能测试...")
        output = vllm_kernel.fused_gptq_gemm_4bit_cuda(
            input_tensor, qweight, qzeros, scales, groupsize
        )
        
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 检查输出非零
        if output.sum().item() == 0:
            print("❌ 输出全零，可能有问题")
            return False
        
        # 检查数值稳定性
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("❌ 输出包含NaN或Inf值")
            return False
        
        print("✅ 功能测试通过!")
        
        # 性能测试
        print("⏱️ 进行性能测试...")
        
        # 预热
        for _ in range(10):
            vllm_kernel.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        
        # 性能测试
        times = []
        for i in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            output = vllm_kernel.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            if i % 20 == 0:
                print(f"🔄 运行 {i+1}/100, 当前延迟: {elapsed_time:.3f}ms")
        
        # 统计结果
        times = np.array(times)
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📊 性能测试结果:")
        print(f"  平均延迟: {avg_time:.3f}ms")
        print(f"  最小延迟: {min_time:.3f}ms")
        print(f"  最大延迟: {max_time:.3f}ms")
        print(f"  标准差: {std_time:.3f}ms")
        
        # 目标评估
        target_time = 0.25
        if avg_time <= target_time:
            print(f"✅ 平均延迟达标! {avg_time:.3f}ms <= {target_time}ms")
        else:
            print(f"⚠️ 平均延迟未达标! {avg_time:.3f}ms > {target_time}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vllm_fixed()
    if success:
        print("\n🎉 vLLM内核修复成功!")
    else:
        print("\n💥 vLLM内核修复失败!")
