#!/usr/bin/env python3
"""
简化的CUDA内核测试 - 在GPU服务器上运行
"""

import torch
import sys
import os

def test_cuda_environment():
    """测试CUDA环境"""
    print("🔍 测试CUDA环境...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU数量: {torch.cuda.device_count()}")
    print(f"🎯 当前GPU: {torch.cuda.current_device()}")
    print(f"📝 GPU名称: {torch.cuda.get_device_name()}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    return True

def test_cuda_kernel_compilation():
    """测试CUDA内核编译"""
    print("\n🔨 测试CUDA内核编译...")
    
    try:
        # 检查CUDA文件是否存在
        cuda_file = "src/gptq_cuda_kernel.cu"
        if not os.path.exists(cuda_file):
            print(f"❌ CUDA文件不存在: {cuda_file}")
            return None
        
        print(f"✅ CUDA文件存在: {cuda_file}")
        
        # 尝试编译 - 使用最简单的参数
        from torch.utils.cpp_extension import load
        
        try:
            print("🔨 尝试编译CUDA内核...")
            fused_gptq_gemm = load(
                name="fused_gptq_gemm_cuda",
                sources=[cuda_file],
                verbose=True
            )
            print("✅ CUDA内核编译成功!")
            return fused_gptq_gemm
        except Exception as e:
            print(f"❌ CUDA内核编译失败: {e}")
            return None
            
    except Exception as e:
        print(f"❌ 编译过程出错: {e}")
        return None

def test_cuda_kernel_execution(fused_gptq_gemm):
    """测试CUDA内核执行"""
    print("\n🧪 测试CUDA内核执行...")
    
    if fused_gptq_gemm is None:
        print("❌ 无法测试执行，内核编译失败")
        return False
    
    try:
        # 创建测试数据
        M, K, N = 1, 4096, 12288  # Qwen7B QKV投影
        groupsize = 128
        num_groups = K // groupsize
        
        # 生成测试数据
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.bfloat16, device='cuda')
        
        print(f"📊 测试数据: input{M}x{K}, qweight{N}x{K//8}, qzeros{num_groups}x{K//8}, scales{num_groups}x{K}")
        
        # 检查可用的函数
        func_names = [attr for attr in dir(fused_gptq_gemm) if not attr.startswith('_')]
        print(f"📋 可用函数: {func_names}")
        
        # 尝试执行CUDA内核
        if hasattr(fused_gptq_gemm, 'fused_gptq_gemm_4bit_cuda'):
            result = fused_gptq_gemm.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        else:
            print("❌ 找不到fused_gptq_gemm_4bit_cuda函数")
            return False
        
        print(f"✅ CUDA内核执行成功!")
        print(f"📊 输出形状: {result.shape}")
        print(f"📊 输出数据类型: {result.dtype}")
        print(f"📊 输出设备: {result.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA内核执行失败: {e}")
        return False

def test_performance(fused_gptq_gemm):
    """测试性能"""
    print("\n⚡ 测试性能...")
    
    if fused_gptq_gemm is None:
        print("❌ 无法测试性能，内核编译失败")
        return
    
    try:
        # 创建测试数据
        M, K, N = 1, 4096, 12288  # Qwen7B QKV投影
        groupsize = 128
        num_groups = K // groupsize
        
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(num_groups, K, dtype=torch.bfloat16, device='cuda')
        
        # 预热
        print("🔥 预热CUDA内核...")
        for _ in range(10):
            _ = fused_gptq_gemm.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
        torch.cuda.synchronize()
        
        # 性能测试
        print("⚡ 性能测试...")
        times = []
        for i in range(100):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = fused_gptq_gemm.fused_gptq_gemm_4bit_cuda(
                input_tensor, qweight, qzeros, scales, groupsize
            )
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            times.append(elapsed)
            
            if i % 20 == 0:
                print(f"  迭代 {i}: {elapsed:.2f}ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        
        # 性能评估
        if avg_time < 0.5:
            print("🎉 性能优秀!")
        elif avg_time < 1.0:
            print("✅ 性能良好")
        elif avg_time < 2.0:
            print("⚠️ 性能一般")
        else:
            print("❌ 性能需要优化")
            
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")

def main():
    """主函数"""
    print("🚀 CUDA内核测试开始...")
    
    # 测试CUDA环境
    if not test_cuda_environment():
        print("❌ CUDA环境测试失败，退出")
        return
    
    # 测试CUDA内核编译
    fused_gptq_gemm = test_cuda_kernel_compilation()
    
    # 测试CUDA内核执行
    if fused_gptq_gemm:
        test_cuda_kernel_execution(fused_gptq_gemm)
        test_performance(fused_gptq_gemm)
    
    print("\n🎉 CUDA内核测试完成!")

if __name__ == "__main__":
    main()