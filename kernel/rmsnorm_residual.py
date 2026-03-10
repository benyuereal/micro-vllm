import torch
import triton
import triton.language as tl
from typing import Tuple
import time


# ============================================================================
# 【原有代码保持不变】
# ============================================================================
@triton.jit
def _rmsnorm_residual_kernel(
        X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res, N, eps, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    R += row_idx * stride_r
    Y += row_idx * stride_y
    RES_OUT += row_idx * stride_res

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
        x_plus_r = x + r
        tl.store(RES_OUT + cols, x_plus_r, mask=mask)
        mean_sq += x_plus_r * x_plus_r

    sum_sq = tl.sum(mean_sq, axis=0)
    mean_sq_scalar = sum_sq / N
    rrms = tl.rsqrt(mean_sq_scalar + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_plus_r = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_plus_r * rrms * w
        tl.store(Y + cols, y, mask=mask)


def rmsnorm_residual_fused(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda, "Input must be on CUDA device"
    assert residual.is_cuda, "Residual must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.shape == residual.shape, f"Shape mismatch: x={x.shape}, residual={residual.shape}"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    residual_flat = residual.view(-1, hidden_dim)
    num_rows = x_flat.shape[0]
    y_flat = torch.empty_like(x_flat)
    res_out_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    _rmsnorm_residual_kernel[(num_rows,)](
        x_flat, residual_flat, y_flat, res_out_flat, weight,
        x_flat.stride(0), residual_flat.stride(0), y_flat.stride(0), res_out_flat.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_flat.view(original_shape), res_out_flat.view(original_shape)


# ============================================================================
# 【修复后的优化版本】
# ============================================================================
@triton.jit
def _rmsnorm_residual_fused_for_gemm_kernel(
        X,  # 输入 x (attn_out)
        R,  # 输入 residual (h)
        Y,  # 预分配的输出 normed
        RES_OUT,  # 预分配的输出 residual
        W,  # ln weight
        stride_x,
        stride_r,
        stride_y,
        stride_res,
        N,  # hidden dim
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    """
    🔧 修复：完全保持原有类型逻辑，只改变内存来源
    """
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    R += row_idx * stride_r
    Y += row_idx * stride_y
    RES_OUT += row_idx * stride_res

    # 第一阶段：和原有代码完全一致
    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
        x_plus_r = x + r

        # 🔧 修复：不做显式类型转换，和原有代码一致
        tl.store(RES_OUT + cols, x_plus_r, mask=mask)
        mean_sq += x_plus_r * x_plus_r

    sum_sq = tl.sum(mean_sq, axis=0)
    mean_sq_scalar = sum_sq / N
    rrms = tl.rsqrt(mean_sq_scalar + eps)

    # 第二阶段：和原有代码完全一致
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_plus_r = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_plus_r * rrms * w

        # 🔧 修复：不做显式类型转换，和原有代码一致
        tl.store(Y + cols, y, mask=mask)


def rmsnorm_residual_fused_for_gemm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out_normed_buffer: torch.Tensor,
        out_residual_buffer: torch.Tensor,
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🔧 修复：确保预分配缓冲区的类型和输入完全一致
    """
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "All must be on CUDA"
    assert out_normed_buffer.is_cuda and out_residual_buffer.is_cuda, "Buffers must be on CUDA"
    assert x.shape == residual.shape, f"Shape mismatch: x={x.shape}, residual={residual.shape}"
    assert x.shape == out_normed_buffer.shape, f"Output buffer shape mismatch"
    assert x.shape == out_residual_buffer.shape, f"Output buffer shape mismatch"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch"

    # 🔧 新增：确保缓冲区类型和输入完全一致
    assert out_normed_buffer.dtype == x.dtype, f"Dtype mismatch: {out_normed_buffer.dtype} vs {x.dtype}"
    assert out_residual_buffer.dtype == x.dtype, f"Dtype mismatch: {out_residual_buffer.dtype} vs {x.dtype}"

    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    residual_flat = residual.view(-1, hidden_dim)
    y_flat = out_normed_buffer.view(-1, hidden_dim)
    res_out_flat = out_residual_buffer.view(-1, hidden_dim)
    num_rows = x_flat.shape[0]

    # 🔧 修复：BLOCK_SIZE 和原有代码保持一致 (8192)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    _rmsnorm_residual_fused_for_gemm_kernel[(num_rows,)](
        x_flat, residual_flat, y_flat, res_out_flat, weight,
        x_flat.stride(0), residual_flat.stride(0), y_flat.stride(0), res_out_flat.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_normed_buffer, out_residual_buffer


def rmsnorm_residual_fused_hybrid(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out_normed_buffer: torch.Tensor,  # 只预分配 normed
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【新增】混合版：只优化 normed 的缓存路径，residual 动态分配
    """
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "All must be on CUDA"
    assert out_normed_buffer.is_cuda, "Buffer must be on CUDA"
    assert x.shape == residual.shape, f"Shape mismatch: x={x.shape}, residual={residual.shape}"
    assert x.shape == out_normed_buffer.shape, f"Output buffer shape mismatch"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch"
    assert out_normed_buffer.dtype == x.dtype, f"Dtype mismatch"

    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    residual_flat = residual.view(-1, hidden_dim)
    y_flat = out_normed_buffer.view(-1, hidden_dim)

    # 🔑 residual 动态分配
    res_out_flat = torch.empty_like(x_flat)

    num_rows = x_flat.shape[0]
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    _rmsnorm_residual_fused_for_gemm_kernel[(num_rows,)](
        x_flat, residual_flat, y_flat, res_out_flat, weight,
        x_flat.stride(0), residual_flat.stride(0), y_flat.stride(0), res_out_flat.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_normed_buffer, res_out_flat.view(original_shape)

# ============================================================================
# 【性能对比测试】
# ============================================================================
def benchmark_rmsnorm_residual():
    # Decode 阶段典型配置
    batch_size = 64
    seq_len = 1
    hidden_size = 4096
    device = "cuda"
    dtype = torch.bfloat16

    # 初始化数据
    attn_out = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    residual_h = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    ln_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    mlp_gu = torch.randn(hidden_size, 2 * hidden_size, device=device, dtype=dtype)

    # 🔧 修复：预分配缓冲区时，确保 dtype 和输入完全一致
    out_normed_buffer = torch.empty_like(attn_out)  # 用 empty_like 继承 dtype
    out_residual_buffer = torch.empty_like(attn_out)
    mlp_out_buffer = torch.empty(batch_size, seq_len, 2 * hidden_size, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # 方案 1: 原有 rmsnorm_residual_fused
    # -------------------------------------------------------------------------
    def original_path():
        normed, res_out = rmsnorm_residual_fused(attn_out, residual_h, ln_weight)
        mlp_out = normed @ mlp_gu
        return mlp_out

    # -------------------------------------------------------------------------
    # 方案 2: 修复后的 rmsnorm_residual_fused_for_gemm
    # -------------------------------------------------------------------------
    def fused_path():
        normed, res_out = rmsnorm_residual_fused_hybrid(
            attn_out, residual_h, ln_weight,
            out_normed_buffer
        )
        torch.matmul(
            normed.view(batch_size, hidden_size),
            mlp_gu,
            out=mlp_out_buffer.view(batch_size, 2 * hidden_size)
        )
        return mlp_out_buffer

    # ============================================================================
    # 预热
    # ============================================================================
    print("Warming up...")
    for _ in range(30):
        original_path()
        fused_path()
    torch.cuda.synchronize()

    # ============================================================================
    # 性能测试
    # ============================================================================
    num_iters = 2000
    print(f"Running benchmark ({num_iters} iterations)...")

    # 测试原有方法
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        original_path()
    torch.cuda.synchronize()
    t1 = time.time()
    original_time = (t1 - t0) / num_iters * 1000  # ms

    # 测试新方法
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        fused_path()
    torch.cuda.synchronize()
    t1 = time.time()
    fused_time = (t1 - t0) / num_iters * 1000  # ms

    # ============================================================================
    # 打印结果
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"🔥 RMSNorm Residual 性能对比 (Decode场景: Batch={batch_size}, Hidden={hidden_size})")
    print("=" * 80)
    print(f"{'方案':<25} | {'耗时 (ms)':<12} | {'相对加速':<10}")
    print("-" * 80)
    print(f"{'原有 rmsnorm_residual_fused':<25} | {original_time:<12.4f} | {'1.00x':<10}")
    print(f"{'修复后 rmsnorm_residual_fused':<25} | {fused_time:<12.4f} | {original_time / fused_time:<10.2f}x")
    print("=" * 80)
    print(f"💡 优化效果：")
    print(f"   - 耗时降低: {(1 - fused_time / original_time) * 100:.1f}%")
    print("=" * 80)

    # 验证正确性
    mlp_original = original_path()
    mlp_fused = fused_path()
    is_correct = torch.allclose(mlp_original, mlp_fused, rtol=1e-3, atol=1e-3)
    print(f"\n✅ 正确性验证: {is_correct}")
    if not is_correct:
        print(f"   最大误差: {torch.max(torch.abs(mlp_original - mlp_fused))}")


if __name__ == "__main__":
    benchmark_rmsnorm_residual()