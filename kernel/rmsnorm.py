"""RMSNorm kernels（Triton）。

一个参数化核心 `_rmsnorm_core`（constexpr 标志 ONE_CENTERED / HAS_RESIDUAL）覆盖
全部变体，公开 API 只有两个函数：
- rmsnorm(x, w, eps, out=None, one_centered=False)：out = x*rrms*w（或 x*rrms*(1+w)）
- rmsnorm_residual(x, r, w, eps, out_normed=None, out_residual=None, one_centered=False)：
  返回 (normed, x+r)

变体维度全部参数化：
- one_centered：False=标准 x*rrms*w（Qwen3/DeepSeek）；True=1-centered x*rrms*(1+w)
  （Qwen3.5，HF Qwen3_5RMSNorm 权重以 0 为中心初始化，1 是隐式 bias）
- residual：rmsnorm_residual 算 norm(x+r) 并输出 x+r（residual 与 normed 用同一 x+r）
- 输出：out=None 分配新 tensor（prefill）；传预分配 buffer 则原地写（decode graph 贴边融合）

TileLang 实验结论（见 /vllm-workspace/tmp/rmsnorm_result.md）：raw kernel 快 ~25% 且
graph-capturable，但 decode 已在 CUDA Graph 内（launch 开销已消），e2e ROI 仅 ~1%；且
TileLang 要求静态 M（prefill 变长 M 不可用）、拒绝 strided buffer（DeepSeek _x16[:bs,0,:]）、
per-M JIT 编译增加 ~64s 启动。故保留 Triton。
"""
import torch
import triton
import triton.language as tl
from typing import Tuple


def _block_size(n: int, cap: int) -> int:
    return min(triton.next_power_of_2(n), cap)


@triton.jit
def _rmsnorm_core(X, R, Y, RES_OUT, W, stride_x, stride_r, stride_y, stride_res, N, eps,
                  ONE_CENTERED: tl.constexpr, HAS_RESIDUAL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """核心 RMSNorm：一行一个 program。
    - HAS_RESIDUAL=False：norm(x)，R/RES_OUT 为 dummy 指针（不读不写）。
    - HAS_RESIDUAL=True：x+=r 存 RES_OUT，norm 读 RES_OUT（保证 residual 与 normed 用同一 x+r）。
    - ONE_CENTERED：权重乘 (1+w) 而非 w。
    两遍：先累加 mean_sq，再写 normed（fp32 计算，bf16 存储）。
    """
    row_idx = tl.program_id(0)
    X += row_idx * stride_x
    Y += row_idx * stride_y
    if HAS_RESIDUAL:
        R += row_idx * stride_r
        RES_OUT += row_idx * stride_res

    mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            tl.store(RES_OUT + cols, x, mask=mask)
        mean_sq += x * x

    rrms = tl.rsqrt(tl.sum(mean_sq, axis=0) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if HAS_RESIDUAL:
            x = tl.load(RES_OUT + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        val = x * rrms * (1.0 + w) if ONE_CENTERED else x * rrms * w
        tl.store(Y + cols, val.to(Y.dtype.element_ty), mask=mask)


def _launch(x, r, y, res_out, w, eps, one_centered, has_residual, block_size):
    """统一 launch：flatten 到 [M,N]，has_residual=False 时 R/RES_OUT 传 y 作 dummy。"""
    n = x.shape[-1]
    x_flat = x.view(-1, n)
    y_flat = y.view(-1, n)
    m = x_flat.shape[0]
    if has_residual:
        r_flat = r.view(-1, n)
        res_flat = res_out.view(-1, n)
    else:
        r_flat = y_flat
        res_flat = y_flat
    _rmsnorm_core[(m,)](
        x_flat, r_flat, y_flat, res_flat, w,
        x_flat.stride(0), r_flat.stride(0), y_flat.stride(0), res_flat.stride(0),
        n, eps, ONE_CENTERED=one_centered, HAS_RESIDUAL=has_residual, BLOCK_SIZE=block_size,
    )


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
            out: torch.Tensor = None, one_centered: bool = False) -> torch.Tensor:
    """RMSNorm：out = x*rrms*w（one_centered=True 时 x*rrms*(1+w)，Qwen3.5）。
    out=None 返回新 tensor（prefill）；传预分配 buffer 则原地写（decode graph 贴边融合）。"""
    if out is None:
        out = torch.empty_like(x)
    _launch(x, None, out, None, weight, eps, one_centered, False, _block_size(x.shape[-1], 8192))
    return out


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                     eps: float = 1e-6, out_normed: torch.Tensor = None,
                     out_residual: torch.Tensor = None,
                     one_centered: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x+residual)：返回 (normed, x+residual)。
    out_*=None 分配新 tensor（prefill）；传预分配 buffer 则原地写（decode graph 贴边融合）。"""
    if out_normed is None:
        out_normed = torch.empty_like(x)
    if out_residual is None:
        out_residual = torch.empty_like(x)
    _launch(x, residual, out_normed, out_residual, weight, eps, one_centered, True,
            _block_size(x.shape[-1], 8192))
    return out_normed, out_residual


# ==================== QK-Norm：融合 qkv buffer 的 q/k 段原地 per-head RMSNorm（Qwen3 专用）====================
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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_size
    x = tl.load(QKV + base + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / head_size + eps)

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
