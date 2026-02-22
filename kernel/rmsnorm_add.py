import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _rmsnorm_kernel(
    X, Y, W, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    Y += row_idx * stride_y

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq += x * x

    sum_sq = tl.sum(mean_sq, axis=0)
    mean_sq_scalar = sum_sq / N
    rrms = tl.rsqrt(mean_sq_scalar + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(Y + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.shape[-1] == weight.shape[0], f"Hidden dim mismatch: {x.shape[-1]} vs {weight.shape[0]}"

    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    num_rows = x_flat.shape[0]
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)

    _rmsnorm_kernel[(num_rows,)](
        x_flat, y_flat, weight, x_flat.stride(0), y_flat.stride(0), hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_flat.view(original_shape)


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


class ProRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"