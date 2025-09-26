#!/usr/bin/env python3
"""
融合LN+QKV GPTQ CUDA内核编译脚本
"""

import os
import torch
from torch.utils.cpp_extension import load

def compile_ln_qkv_fusion_kernel():
    """编译融合LN+QKV GPTQ CUDA内核"""
    print("🚀 融合LN+QKV GPTQ CUDA内核编译开始...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法编译CUDA内核")
    
    cuda_version = torch.version.cuda
    print(f"检测到CUDA版本: {cuda_version}")
    
    # 融合LN+QKV文件
    fusion_file = "ln_qkv_fusion_kernel.cu"
    if not os.path.exists(fusion_file):
        print(f"❌ 融合LN+QKV CUDA文件不存在: {fusion_file}")
        return None
    
    print(f"✅ 融合LN+QKV CUDA文件存在: {fusion_file}")
    
    # 优化编译选项
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-Xptxas=-O3",
        "--ptxas-options=-v",
        "-maxrregcount=255",
        "-lineinfo",
        "-std=c++17"
    ]
    
    # 编译融合LN+QKV内核
    try:
        print("🔨 开始编译融合LN+QKV CUDA内核...")
        fused_ln_qkv_gptq = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=[fusion_file],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False
        )
        print("✅ 融合LN+QKV CUDA内核编译成功!")
        return fused_ln_qkv_gptq
    except Exception as e:
        print(f"❌ 融合LN+QKV CUDA内核编译失败: {e}")
        return None

def test_ln_qkv_fusion_kernel():
    """测试融合LN+QKV内核功能"""
    print("\n🧪 测试融合LN+QKV内核功能...")
    
    try:
        # 导入编译的内核
        import fused_ln_qkv_gptq_cuda
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        num_heads, kv_num_heads, head_size = 32, 32, 128
        groupsize = 128
        num_groups = hidden_dim // groupsize
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        
        # 输出张量
        q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        
        print(f"📊 测试数据: input{batch_size}x{seq_len}x{hidden_dim}")
        print(f"📊 GPTQ参数: qweight{qweight.shape}, qzeros{qzeros.shape}, scales{scales.shape}")
        
        # 功能测试
        fused_ln_qkv_gptq_cuda.fused_ln_qkv_gptq_cuda(
            input_tensor.flatten(),
            qweight,
            qzeros,
            scales,
            ln_weight,
            ln_bias,
            q_output.flatten(),
            k_output.flatten(),
            v_output.flatten(),
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            kv_num_heads,
            head_size,
            groupsize,
            1e-6
        )
        
        print("✅ 融合LN+QKV内核功能测试成功!")
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        
        # 验证输出形状
        expected_shape = (batch_size, num_heads, seq_len, head_size)
        assert q_output.shape == expected_shape, f"Q形状错误: {q_output.shape} != {expected_shape}"
        assert k_output.shape == expected_shape, f"K形状错误: {k_output.shape} != {expected_shape}"
        assert v_output.shape == expected_shape, f"V形状错误: {v_output.shape} != {expected_shape}"
        
        # 检查输出是否包含有效值
        if torch.any(torch.isnan(q_output)) or torch.any(torch.isinf(q_output)):
            print("❌ Q输出包含无效值")
            return False
        
        if torch.any(torch.isnan(k_output)) or torch.any(torch.isinf(k_output)):
            print("❌ K输出包含无效值")
            return False
        
        if torch.any(torch.isnan(v_output)) or torch.any(torch.isinf(v_output)):
            print("❌ V输出包含无效值")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 融合LN+QKV内核功能测试失败: {e}")
        return False

def benchmark_ln_qkv_fusion_kernel():
    """融合LN+QKV内核性能测试"""
    print("\n⚡ 融合LN+QKV内核性能测试...")
    
    try:
        import fused_ln_qkv_gptq_cuda
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        num_heads, kv_num_heads, head_size = 32, 32, 128
        groupsize = 128
        num_groups = hidden_dim // groupsize
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        
        # 输出张量
        q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda')
        
        # 预热
        for _ in range(10):
            fused_ln_qkv_gptq_cuda.fused_ln_qkv_gptq_cuda(
                input_tensor.flatten(),
                qweight,
                qzeros,
                scales,
                ln_weight,
                ln_bias,
                q_output.flatten(),
                k_output.flatten(),
                v_output.flatten(),
                batch_size,
                seq_len,
                hidden_dim,
                num_heads,
                kv_num_heads,
                head_size,
                groupsize,
                1e-6
            )
        torch.cuda.synchronize()
        
        # 性能测试
        num_runs = 50
        timings = []
        
        for i in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            fused_ln_qkv_gptq_cuda.fused_ln_qkv_gptq_cuda(
                input_tensor.flatten(),
                qweight,
                qzeros,
                scales,
                ln_weight,
                ln_bias,
                q_output.flatten(),
                k_output.flatten(),
                v_output.flatten(),
                batch_size,
                seq_len,
                hidden_dim,
                num_heads,
                kv_num_heads,
                head_size,
                groupsize,
                1e-6
            )
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
            
            if i % 10 == 0:
                print(f"  迭代 {i}: {timings[-1]:.2f}ms")
        
        avg_time = sum(timings) / num_runs
        min_time = min(timings)
        max_time = max(timings)
        
        print(f"\n📊 融合LN+QKV内核性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最小时间: {min_time:.2f}ms")
        print(f"  最大时间: {max_time:.2f}ms")
        
        # 性能评估
        target_time = 0.25  # 目标时间: 0.25ms
        if avg_time < target_time:
            print("🎉 融合LN+QKV内核达到目标性能!")
        elif avg_time < target_time * 2:
            print("✅ 融合LN+QKV内核接近目标性能")
        elif avg_time < target_time * 5:
            print("⚠️ 融合LN+QKV内核需要进一步优化")
        else:
            print("❌ 融合LN+QKV内核性能不达标")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ 融合LN+QKV内核性能测试失败: {e}")
        return None

def compare_with_separate_operators():
    """与分离算子性能对比"""
    print("\n🔄 与分离算子性能对比...")
    
    try:
        import fused_ln_qkv_gptq_cuda
        from core.layer.gptq import GPTQCUDAFusion
        
        # 测试数据
        batch_size, seq_len, hidden_dim = 1, 1, 4096
        num_heads, kv_num_heads, head_size = 32, 32, 128
        groupsize = 128
        num_groups = hidden_dim // groupsize
        
        # 输入数据
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, device='cuda')
        
        # GPTQ参数
        qweight = torch.randint(0, 256, (hidden_dim // 8, hidden_dim * 3), dtype=torch.uint32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, groupsize // 8), dtype=torch.uint32, device='cuda')
        scales = torch.randn(num_groups, hidden_dim * 3, dtype=torch.float16, device='cuda')
        
        # LayerNorm参数
        ln_weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        ln_bias = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        
        # 融合算子性能测试
        print("📊 测试融合算子性能...")
        fused_timings = []
        
        for _ in range(50):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            fused_ln_qkv_gptq_cuda.fused_ln_qkv_gptq_cuda(
                input_tensor.flatten(),
                qweight,
                qzeros,
                scales,
                ln_weight,
                ln_bias,
                torch.zeros(batch_size, num_heads, seq_len, head_size, dtype=torch.float16, device='cuda').flatten(),
                torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda').flatten(),
                torch.zeros(batch_size, kv_num_heads, seq_len, head_size, dtype=torch.float16, device='cuda').flatten(),
                batch_size,
                seq_len,
                hidden_dim,
                num_heads,
                kv_num_heads,
                head_size,
                groupsize,
                1e-6
            )
            end_event.record()
            torch.cuda.synchronize()
            fused_timings.append(start_event.elapsed_time(end_event))
        
        fused_avg = sum(fused_timings) / len(fused_timings)
        
        # 分离算子性能测试
        print("📊 测试分离算子性能...")
        gptq_fusion = GPTQCUDAFusion()
        
        separate_timings = []
        for _ in range(50):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            # LayerNorm
            ln_output = torch.nn.functional.layer_norm(
                input_tensor, 
                input_tensor.shape[-1:], 
                ln_weight, 
                ln_bias
            )
            
            # QKV投影
            batch_size, seq_len, hidden_dim = ln_output.shape
            input_2d = ln_output.view(-1, hidden_dim)
            
            qkv_output = gptq_fusion.fused_gptq_gemm_4bit(
                input=input_2d,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales
            )
            
            # 重塑和分割
            qkv_output = qkv_output.view(batch_size, seq_len, -1)
            hidden_size = qkv_output.shape[-1] // 3
            q, k, v = qkv_output.split(hidden_size, dim=-1)
            
            # 重塑为head格式
            q = q.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, kv_num_heads, head_size).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, kv_num_heads, head_size).permute(0, 2, 1, 3)
            
            end_event.record()
            torch.cuda.synchronize()
            separate_timings.append(start_event.elapsed_time(end_event))
        
        separate_avg = sum(separate_timings) / len(separate_timings)
        speedup = separate_avg / fused_avg
        
        print(f"\n📊 性能对比结果:")
        print(f"  融合算子平均时间: {fused_avg:.2f}ms")
        print(f"  分离算子平均时间: {separate_avg:.2f}ms")
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 2.0:
            print("🎉 融合算子显著提升性能!")
        elif speedup > 1.5:
            print("✅ 融合算子提升性能")
        else:
            print("⚠️ 融合算子性能提升有限")
        
        return speedup
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 融合LN+QKV GPTQ CUDA内核编译和测试")
    print("=" * 60)
    
    # 编译内核
    kernel = compile_ln_qkv_fusion_kernel()
    if kernel is None:
        print("❌ 编译失败，退出")
        exit(1)
    
    # 功能测试
    if not test_ln_qkv_fusion_kernel():
        print("❌ 功能测试失败，退出")
        exit(1)
    
    # 性能测试
    avg_time = benchmark_ln_qkv_fusion_kernel()
    if avg_time is None:
        print("❌ 性能测试失败，退出")
        exit(1)
    
    # 性能对比
    speedup = compare_with_separate_operators()
    if speedup is None:
        print("❌ 性能对比测试失败，退出")
        exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 融合LN+QKV GPTQ CUDA内核测试完成!")
    print(f"📊 最终性能: {avg_time:.2f}ms")
    print(f"📊 加速比: {speedup:.2f}x")
    print("=" * 60)
