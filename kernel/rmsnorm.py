import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _rmsnorm_kernel(X, Y, W, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    Y += row_idx * stride_y

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq += x * x

    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, x * rrms * w, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm_kernel[(x_flat.shape[0],)](
        x_flat, y_flat, weight, x_flat.stride(0), y_flat.stride(0), hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_flat.view(original_shape)


@triton.jit
def _rmsnorm_residual_kernel(X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res, N, eps,
                             BLOCK_SIZE: tl.constexpr):
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

    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_plus_r = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, x_plus_r * rrms * w, mask=mask)


def rmsnorm_residual_fused(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                           eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x+residual)：返回 (normed, x+residual)。prefill 路径用。"""
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    r_flat = residual.view(-1, hidden_dim)
    y_flat = torch.empty_like(x_flat)
    res_out = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm_residual_kernel[(x_flat.shape[0],)](
        x_flat, r_flat, y_flat, res_out, weight,
        x_flat.stride(0), r_flat.stride(0), y_flat.stride(0), res_out.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_flat.view(original_shape), res_out.view(original_shape)


# ---- 贴边融合版：norm 结果直接写预分配 buffer，省一次 copy（decode graph 路径用）----
@triton.jit
def _rmsnorm_gemm_kernel(X, Y, W, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    Y += row_idx * stride_y

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq += x * x

    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (x * rrms * w).to(tl.bfloat16), mask=mask)


def rmsnorm_(x: torch.Tensor, weight: torch.Tensor, out_buffer: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm 结果直接写入 out_buffer（matmul 输入），decode 贴边融合用。"""
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    y_flat = out_buffer.view(-1, hidden_dim)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 2048)
    _rmsnorm_gemm_kernel[(x_flat.shape[0],)](
        x_flat, y_flat, weight, x_flat.stride(0), y_flat.stride(0), hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_buffer


@triton.jit
def _rmsnorm_residual_fused_for_gemm_kernel(X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res,
                                            N, eps, BLOCK_SIZE: tl.constexpr):
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

    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_plus_r = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, x_plus_r * rrms * w, mask=mask)


def rmsnorm_residual_gemm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                          out_normed_buffer: torch.Tensor, out_residual_buffer: torch.Tensor,
                          eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x+residual) 贴边融合版：normed 与 residual 均写预分配 buffer，decode graph 路径用。"""
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    r_flat = residual.view(-1, hidden_dim)
    y_flat = out_normed_buffer.view(-1, hidden_dim)
    res_out = out_residual_buffer.view(-1, hidden_dim)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm_residual_fused_for_gemm_kernel[(x_flat.shape[0],)](
        x_flat, r_flat, y_flat, res_out, weight,
        x_flat.stride(0), r_flat.stride(0), y_flat.stride(0), res_out.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_normed_buffer, out_residual_buffer
