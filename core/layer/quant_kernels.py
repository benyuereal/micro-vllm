# File: quant_kernels.py
import sys

import torch
import triton
import triton.language as tl
import numpy as np
import time


class QuantKernels:
    """
    INT4 GPTQ 量化内核 - Qwen7B 专用
    专注消除量化模型的反量化开销
    """

    @staticmethod
    def fused_quant_qkv_proj(
            hidden_states: torch.Tensor,
            qkv_weight: torch.Tensor,
            qkv_scale: torch.Tensor,
            qkv_zero: torch.Tensor,
            num_heads: int,
            head_dim: int,
            group_size: int = 128
    ) -> tuple:
        """
        融合反量化的 QKV 投影内核 (Qwen7B 专用)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 创建输出张量 [B, H, S, D]
        q = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)
        k = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)
        v = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)

        # 设置 Triton 网格
        grid = (batch_size, triton.cdiv(seq_len, 16))

        # 启动量化内核
        QuantKernels._fused_quant_qkv_proj_kernel[grid](
            hidden_states,
            qkv_weight,
            qkv_scale,
            qkv_zero,
            q, k, v,
            batch_size, seq_len, hidden_dim,
            num_heads, head_dim,
            group_size,
            BLOCK_M=16,
            BLOCK_K=32
        )

        return q, k, v

    @staticmethod
    def fused_quant_out_proj(
            attn_output: torch.Tensor,
            out_weight: torch.Tensor,
            out_scale: torch.Tensor,
            out_zero: torch.Tensor,
            group_size: int = 128
    ) -> torch.Tensor:
        """
        融合反量化的输出投影内核 (Qwen7B 专用)
        """
        batch_size, seq_len, hidden_dim = attn_output.shape

        # 创建输出张量
        output = torch.empty_like(attn_output)

        # 设置 Triton 网格
        grid = (batch_size, triton.cdiv(seq_len, 16))

        # 启动量化内核
        QuantKernels._fused_quant_out_proj_kernel[grid](
            attn_output,
            out_weight,
            out_scale,
            out_zero,
            output,
            batch_size, seq_len, hidden_dim,
            group_size,
            BLOCK_M=16,
            BLOCK_K=32
        )

        return output

    @staticmethod
    @triton.jit
    def _fused_quant_qkv_proj_kernel(
            # 输入指针
            hidden_states_ptr, qkv_weight_ptr, qkv_scale_ptr, qkv_zero_ptr,
            q_ptr, k_ptr, v_ptr,
            batch_size, seq_len, hidden_dim, num_heads, head_dim,
            group_size: tl.constexpr,
            BLOCK_M: tl.constexpr = 16,
            BLOCK_K: tl.constexpr = 32
    ):
        # 计算 PID 和偏移
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        # 计算输入偏移 [B, S, D]
        input_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim

        # 计算输出偏移 [B, H, S, D]
        q_offset = pid_b * num_heads * seq_len * head_dim + pid_s * BLOCK_M * head_dim

        # 初始化累加器
        acc_q = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        acc_k = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

        # 加载输入块 [BLOCK_M, hidden_dim]
        input_block = tl.load(
            hidden_states_ptr + input_offset,
            shape=(BLOCK_M, hidden_dim),
            other=0.0
        )

        # 循环处理 K 维度（分组处理）
        for k in range(0, hidden_dim, BLOCK_K):
            # 加载量化权重块 [hidden_dim, 3*hidden_dim]
            weight_offset = k * 3 * hidden_dim
            quant_weight = tl.load(
                qkv_weight_ptr + weight_offset,
                shape=(hidden_dim, 3 * hidden_dim),
                other=0
            )

            # 加载量化参数
            group_idx = k // group_size
            scale = tl.load(qkv_scale_ptr + group_idx)
            zero = tl.load(qkv_zero_ptr + group_idx)

            # 反量化权重 (INT8 -> FP32)
            weight_fp32 = (quant_weight.to(tl.float32) - zero) * scale

            # 分割 QKV 权重
            q_weight = weight_fp32[:, :hidden_dim]
            k_weight = weight_fp32[:, hidden_dim:2 * hidden_dim]
            v_weight = weight_fp32[:, 2 * hidden_dim:3 * hidden_dim]

            # 矩阵乘法
            acc_q += tl.dot(input_block, q_weight)
            acc_k += tl.dot(input_block, k_weight)
            acc_v += tl.dot(input_block, v_weight)

        # 存储输出 [BLOCK_M, head_dim]
        tl.store(q_ptr + q_offset, acc_q)
        tl.store(k_ptr + q_offset, acc_k)
        tl.store(v_ptr + q_offset, acc_v)

    @staticmethod
    @triton.jit
    def _fused_quant_out_proj_kernel(
            # 输入指针
            attn_output_ptr, out_weight_ptr, out_scale_ptr, out_zero_ptr,
            out_ptr, batch_size, seq_len, hidden_dim, group_size: tl.constexpr,
            BLOCK_M: tl.constexpr = 16,
            BLOCK_K: tl.constexpr = 32
    ):
        # 计算 PID 和偏移
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        # 计算输入偏移 [B, S, D]
        input_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim

        # 计算输出偏移 [B, S, D]
        out_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim

        # 初始化累加器
        acc = tl.zeros((BLOCK_M, hidden_dim), dtype=tl.float32)

        # 加载输入块 [BLOCK_M, hidden_dim]
        input_block = tl.load(
            attn_output_ptr + input_offset,
            shape=(BLOCK_M, hidden_dim),
            other=0.0
        )

        # 循环处理 K 维度（分组处理）
        for k in range(0, hidden_dim, BLOCK_K):
            # 加载量化权重块 [hidden_dim, hidden_dim]
            weight_offset = k * hidden_dim
            quant_weight = tl.load(
                out_weight_ptr + weight_offset,
                shape=(hidden_dim, hidden_dim),
                other=0
            )

            # 加载量化参数
            group_idx = k // group_size
            scale = tl.load(out_scale_ptr + group_idx)
            zero = tl.load(out_zero_ptr + group_idx)

            # 反量化权重 (INT8 -> FP32)
            weight_fp32 = (quant_weight.to(tl.float32) - zero) * scale

            # 矩阵乘法
            acc += tl.dot(input_block, weight_fp32)

        # 存储输出 [BLOCK_M, hidden_dim]
        tl.store(out_ptr + out_offset, acc)


# ==================== 测试方法 ====================

def test_fused_quant_qkv_proj():
    """
    测试融合量化QKV投影内核
    """
    print("\n" + "=" * 60)
    print("Testing fused_quant_qkv_proj...")

    # 设置测试参数
    batch_size, seq_len, hidden_dim = 2, 128, 4096
    num_heads, head_dim = 32, 128
    group_size = 128

    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim,
                                device="cuda", dtype=torch.float16)

    # 创建模拟量化权重 (INT8 范围)
    qkv_weight = torch.randint(-128, 128, (hidden_dim, 3 * hidden_dim),
                               device="cuda", dtype=torch.int8)
    qkv_scale = torch.ones(hidden_dim // group_size, device="cuda", dtype=torch.float16)
    qkv_zero = torch.zeros(hidden_dim // group_size, device="cuda", dtype=torch.float16)

    try:
        # 执行融合内核
        q, k, v = QuantKernels.fused_quant_qkv_proj(
            hidden_states=hidden_states,
            qkv_weight=qkv_weight,
            qkv_scale=qkv_scale,
            qkv_zero=qkv_zero,
            num_heads=num_heads,
            head_dim=head_dim,
            group_size=group_size
        )

        # 验证输出形状
        assert q.shape == (batch_size, num_heads, seq_len, head_dim)
        assert k.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v.shape == (batch_size, num_heads, seq_len, head_dim)

        # 验证输出类型
        assert q.dtype == torch.float16
        assert k.dtype == torch.float16
        assert v.dtype == torch.float16

        # 验证输出值范围
        assert not torch.isnan(q).any(), "Q contains NaN values"
        assert not torch.isinf(q).any(), "Q contains Inf values"

        print("✅ fused_quant_qkv_proj test passed!")
        print(f"   Output shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        return True

    except Exception as e:
        print(f"❌ fused_quant_qkv_proj test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fused_quant_out_proj():
    """
    测试融合量化输出投影内核
    """
    print("\n" + "=" * 60)
    print("Testing fused_quant_out_proj...")

    # 设置测试参数
    batch_size, seq_len, hidden_dim = 2, 128, 4096
    group_size = 128

    # 创建测试数据
    attn_output = torch.randn(batch_size, seq_len, hidden_dim,
                              device="cuda", dtype=torch.float16)

    # 创建模拟量化权重 (INT8 范围)
    out_weight = torch.randint(-128, 128, (hidden_dim, hidden_dim),
                               device="cuda", dtype=torch.int8)
    out_scale = torch.ones(hidden_dim // group_size, device="cuda", dtype=torch.float16)
    out_zero = torch.zeros(hidden_dim // group_size, device="cuda", dtype=torch.float16)

    try:
        # 执行融合内核
        output = QuantKernels.fused_quant_out_proj(
            attn_output=attn_output,
            out_weight=out_weight,
            out_scale=out_scale,
            out_zero=out_zero,
            group_size=group_size
        )

        # 验证输出
        assert output.shape == attn_output.shape
        assert output.dtype == torch.float16

        # 验证输出值范围
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        print("✅ fused_quant_out_proj test passed!")
        print(f"   Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"❌ fused_quant_out_proj test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """
    测试量化内核 vs 原始实现的性能对比
    """
    print("\n" + "=" * 60)
    print("Testing performance comparison...")

    # 设置测试参数
    batch_size, seq_len, hidden_dim = 1, 2048, 4096
    num_heads, head_dim = 32, 128
    group_size = 128
    warmup_iters, test_iters = 5, 20

    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim,
                                device="cuda", dtype=torch.float16)

    # 创建模拟量化权重
    qkv_weight = torch.randint(-128, 128, (hidden_dim, 3 * hidden_dim),
                               device="cuda", dtype=torch.int8)
    qkv_scale = torch.ones(hidden_dim // group_size, device="cuda", dtype=torch.float16)
    qkv_zero = torch.zeros(hidden_dim // group_size, device="cuda", dtype=torch.float16)

    # 测试融合内核性能
    torch.cuda.synchronize()
    fused_times = []

    for i in range(warmup_iters + test_iters):
        start_time = time.time()

        q, k, v = QuantKernels.fused_quant_qkv_proj(
            hidden_states=hidden_states,
            qkv_weight=qkv_weight,
            qkv_scale=qkv_scale,
            qkv_zero=qkv_zero,
            num_heads=num_heads,
            head_dim=head_dim,
            group_size=group_size
        )

        torch.cuda.synchronize()
        if i >= warmup_iters:
            fused_times.append(time.time() - start_time)

    avg_fused_time = np.mean(fused_times) * 1000
    std_fused_time = np.std(fused_times) * 1000

    # 测试原始实现性能 (模拟反量化)
    def original_implementation():
        # 反量化
        qkv_weight_fp16 = (qkv_weight.float() - qkv_zero[None, :].float()) * qkv_scale[None, :].float()

        # 分割QKV权重
        q_weight = qkv_weight_fp16[:, :hidden_dim]
        k_weight = qkv_weight_fp16[:, hidden_dim:2 * hidden_dim]
        v_weight = qkv_weight_fp16[:, 2 * hidden_dim:3 * hidden_dim]

        # 矩阵乘法
        q = hidden_states @ q_weight
        k = hidden_states @ k_weight
        v = hidden_states @ v_weight

        # 重塑为 [B, H, S, D]
        q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        return q, k, v

    original_times = []
    for i in range(warmup_iters + test_iters):
        start_time = time.time()
        q, k, v = original_implementation()
        torch.cuda.synchronize()
        if i >= warmup_iters:
            original_times.append(time.time() - start_time)

    avg_original_time = np.mean(original_times) * 1000
    std_original_time = np.std(original_times) * 1000

    # 计算加速比
    speedup = avg_original_time / avg_fused_time

    print("✅ Performance comparison completed!")
    print(f"   Original implementation: {avg_original_time:.2f} ± {std_original_time:.2f} ms")
    print(f"   Fused kernel: {avg_fused_time:.2f} ± {std_fused_time:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")

    return avg_original_time, avg_fused_time, speedup


def test_correctness_comparison():
    """
    测试融合内核与原始实现的数值一致性
    """
    print("\n" + "=" * 60)
    print("Testing correctness comparison...")

    # 设置测试参数
    batch_size, seq_len, hidden_dim = 1, 64, 4096
    num_heads, head_dim = 32, 128
    group_size = 128

    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim,
                                device="cuda", dtype=torch.float16)

    # 创建模拟量化权重
    qkv_weight = torch.randint(-128, 128, (hidden_dim, 3 * hidden_dim),
                               device="cuda", dtype=torch.int8)
    qkv_scale = torch.ones(hidden_dim // group_size, device="cuda", dtype=torch.float16)
    qkv_zero = torch.zeros(hidden_dim // group_size, device="cuda", dtype=torch.float16)

    try:
        # 执行融合内核
        q_fused, k_fused, v_fused = QuantKernels.fused_quant_qkv_proj(
            hidden_states=hidden_states,
            qkv_weight=qkv_weight,
            qkv_scale=qkv_scale,
            qkv_zero=qkv_zero,
            num_heads=num_heads,
            head_dim=head_dim,
            group_size=group_size
        )

        # 执行原始实现
        def original_implementation():
            # 反量化
            qkv_weight_fp16 = (qkv_weight.float() - qkv_zero[None, :].float()) * qkv_scale[None, :].float()

            # 分割QKV权重
            q_weight = qkv_weight_fp16[:, :hidden_dim]
            k_weight = qkv_weight_fp16[:, hidden_dim:2 * hidden_dim]
            v_weight = qkv_weight_fp16[:, 2 * hidden_dim:3 * hidden_dim]

            # 矩阵乘法
            q = hidden_states @ q_weight
            k = hidden_states @ k_weight
            v = hidden_states @ v_weight

            # 重塑为 [B, H, S, D]
            q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            return q, k, v

        q_orig, k_orig, v_orig = original_implementation()

        # 转换为float32进行比较
        q_fused = q_fused.float()
        k_fused = k_fused.float()
        v_fused = v_fused.float()
        q_orig = q_orig.float()
        k_orig = k_orig.float()
        v_orig = v_orig.float()

        # 计算相对误差
        q_error = torch.norm(q_fused - q_orig) / torch.norm(q_orig)
        k_error = torch.norm(k_fused - k_orig) / torch.norm(k_orig)
        v_error = torch.norm(v_fused - v_orig) / torch.norm(v_orig)

        # 检查是否在合理范围内
        tolerance = 1e-3
        assert q_error < tolerance, f"Q relative error too large: {q_error.item()}"
        assert k_error < tolerance, f"K relative error too large: {k_error.item()}"
        assert v_error < tolerance, f"V relative error too large: {v_error.item()}"

        print("✅ Correctness comparison passed!")
        print(f"   Q relative error: {q_error.item():.6f}")
        print(f"   K relative error: {k_error.item():.6f}")
        print(f"   V relative error: {v_error.item():.6f}")

        return True

    except Exception as e:
        print(f"❌ Correctness comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 主入口 ====================

def main():
    """
    主测试入口
    """
    print("Starting Quant Kernel Tests")
    print("=" * 60)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Tests require GPU.")
        return False

    # 设置设备
    torch.cuda.set_device(0)

    # 禁用梯度计算
    with torch.no_grad():
        # 运行所有测试
        tests = [
            test_fused_quant_qkv_proj,
            test_fused_quant_out_proj,
            test_performance_comparison,
            test_correctness_comparison
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
                import traceback
                traceback.print_exc()
                results.append(False)

        # 打印总结
        print("\n" + "=" * 60)
        print("Test Summary:")
        print("=" * 60)

        for i, (test, result) in enumerate(zip(tests, results)):
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{i + 1}. {test.__name__}: {status}")

        all_passed = all(results)
        print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

        return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)