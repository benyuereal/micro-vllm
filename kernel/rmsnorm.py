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


# ---- 1-centered RMSNorm（Qwen3.5 专用）：out = x * rrms * (1 + w) ----
# HF Qwen3_5RMSNorm: output = _norm(x.float()) * (1.0 + weight.float())，与 Qwen3 的
# x * w 不同（权重以 0 为中心初始化，1 是隐式 bias）。
@triton.jit
def _rmsnorm1_kernel(X, Y, W, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr):
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
        tl.store(Y + cols, (x * rrms * (1.0 + w)).to(Y.dtype.element_ty), mask=mask)


def rmsnorm1(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1-centered RMSNorm（Qwen3.5）：out = x * rrms * (1 + w)。"""
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    y_flat = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm1_kernel[(x_flat.shape[0],)](
        x_flat, y_flat, weight, x_flat.stride(0), y_flat.stride(0), hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_flat.view(original_shape)


@triton.jit
def _rmsnorm1_residual_kernel(X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res, N, eps,
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
        tl.store(Y + cols, (x_plus_r * rrms * (1.0 + w)).to(Y.dtype.element_ty), mask=mask)


def rmsnorm1_residual_fused(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                            eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-centered RMSNorm(x+residual)：返回 (normed, x+residual)。prefill 路径用。"""
    original_shape = x.shape
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    r_flat = residual.view(-1, hidden_dim)
    y_flat = torch.empty_like(x_flat)
    res_out = torch.empty_like(x_flat)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm1_residual_kernel[(x_flat.shape[0],)](
        x_flat, r_flat, y_flat, res_out, weight,
        x_flat.stride(0), r_flat.stride(0), y_flat.stride(0), res_out.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_flat.view(original_shape), res_out.view(original_shape)


@triton.jit
def _rmsnorm1_gemm_kernel(X, Y, W, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr):
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
        tl.store(Y + cols, (x * rrms * (1.0 + w)).to(tl.bfloat16), mask=mask)


def rmsnorm1_(x: torch.Tensor, weight: torch.Tensor, out_buffer: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1-centered RMSNorm 结果直接写入 out_buffer（decode 贴边融合用）。"""
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    y_flat = out_buffer.view(-1, hidden_dim)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 2048)
    _rmsnorm1_gemm_kernel[(x_flat.shape[0],)](
        x_flat, y_flat, weight, x_flat.stride(0), y_flat.stride(0), hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_buffer


@triton.jit
def _rmsnorm1_residual_gemm_kernel(X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res,
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
        tl.store(Y + cols, (x_plus_r * rrms * (1.0 + w)).to(tl.bfloat16), mask=mask)


def rmsnorm1_residual_gemm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                           out_normed_buffer: torch.Tensor, out_residual_buffer: torch.Tensor,
                           eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-centered RMSNorm(x+residual) 贴边融合版：decode graph 路径用。"""
    hidden_dim = x.shape[-1]
    x_flat = x.view(-1, hidden_dim)
    r_flat = residual.view(-1, hidden_dim)
    y_flat = out_normed_buffer.view(-1, hidden_dim)
    res_out = out_residual_buffer.view(-1, hidden_dim)
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 8192)
    _rmsnorm1_residual_gemm_kernel[(x_flat.shape[0],)](
        x_flat, r_flat, y_flat, res_out, weight,
        x_flat.stride(0), r_flat.stride(0), y_flat.stride(0), res_out.stride(0),
        hidden_dim, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_normed_buffer, out_residual_buffer


# ---- QK-Norm：对融合 qkv buffer 的 q 段/k 段原地 per-head RMSNorm（Qwen3 专用）----
@triton.jit
def _qk_norm_kernel(QKV, W, stride_qkv_row, seg_offset, head_size: tl.constexpr,
                    num_heads: tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    """每个 program 处理一个 (batch, head)。
    pid = batch_idx * num_heads + head_idx
    该 head 在 qkv_buf 中的起始 = batch_idx*stride_qkv_row + seg_offset + head_idx*head_size
    两遍：先算 mean_sq 再写归一结果（原地安全，同 program 内顺序执行）。
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    base = batch_idx * stride_qkv_row + seg_offset + head_idx * head_size

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_size
    x = tl.load(QKV + base + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = x * x
    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / head_size + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(QKV + base + cols, (x * rrms * w).to(tl.bfloat16), mask=mask)


def qk_norm_inplace(qkv_buf: torch.Tensor, bs: int, q_dim: int, kv_dim: int,
                    q_weight: torch.Tensor, k_weight: torch.Tensor,
                    num_heads: int, kv_num_heads: int, head_size: int,
                    eps: float = 1e-6):
    """对 [max_bs, q_dim+2*kv_dim] 融合 qkv buffer 的 q 段、k 段原地做 per-head RMSNorm。

    decode graph 路径用：直接在 qkv_buf 上原地 norm，无需额外 buffer，无需 reshape（支持
    qkv_buf 行 stride != q_dim 的非连续情况）。替代旧 PyTorch 原生 op 的 ~6 个碎片 kernel。
    """
    BLOCK_SIZE = triton.next_power_of_2(head_size)
    # q 段：seg_offset=0, num_heads
    _qk_norm_kernel[(bs * num_heads,)](
        qkv_buf, q_weight, qkv_buf.stride(0), 0, head_size, num_heads, eps, BLOCK_SIZE=BLOCK_SIZE)
    # k 段：seg_offset=q_dim, kv_num_heads
    _qk_norm_kernel[(bs * kv_num_heads,)](
        qkv_buf, k_weight, qkv_buf.stride(0), q_dim, head_size, kv_num_heads, eps, BLOCK_SIZE=BLOCK_SIZE)
