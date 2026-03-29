"""
W8A16 动态反量化 GEMM（per-channel 对称量化）

反量化（kernel 内部）：
  B_fp[k, n] ≈ B_int8[k, n] * scale[n]

收益：
  weights 从 HBM 读取量减少 50%（INT8 vs BF16），
  对 decode 场景（memory-bound）有直接带宽收益。
  反量化只需 int→fp 类型转换 + 一次 elementwise mul，
  在寄存器里完成，不走 HBM，几乎无额外延迟。

接口：
  matmul_dequant(a, b_int8, scale) → c
    a:      [M, K]  BF16/FP16
    b_int8: [K, N]  torch.int8  （由 torch.quantize_per_channel 产出）
    scale:  [N]     BF16/FP16   per-channel dequant scale
    c:      [M, N]  同 a 的 dtype

量化（调用方负责，推荐用 torch 原生 API）：
  b_q    = torch.quantize_per_channel(
               b_fp.flvoat(), scales, zero_points, axis=1, dtype=torch.qint8)
  b_int8 = b_q.int_repr()
  scale  = b_q.q_per_channel_scales().to(b_fp.dtype)
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from matmul import matmul_v3


_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.autotune(
    configs=[
        # INT8 weight 只占 1 byte，BLOCK_K 可以翻倍，shared mem 用量与 BF16 相当
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_dequant_kernel(
        a_ptr, b_ptr, scale_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr = tl.bfloat16,
):
    """
    A: [M, K]  FP16/BF16  activations
    B: [K, N]  INT8       weights (per-channel quantized, symmetric)
    scale: [N] FP16/BF16  per-channel dequant scale
    C: [M, N]  FP16/BF16  output
    """
    BLOCK_M: tl.constexpr = 32  # 覆盖全部 M 行（decode M ≤ 32）

    pid_n = tl.program_id(axis=0)

    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # ── per-channel scale：在 K 循环外加载一次，整个 tile 复用 ────────────
    # shape [BLOCK_N]，广播到 [BLOCK_K, BLOCK_N] 用于反量化
    scale = tl.load(scale_ptr + rn, mask=rn < N, other=1.0)

    m_mask = rm[:, None] < M
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a    = tl.load(a_ptrs, mask=m_mask, other=0.0)         # M mask 防止 M=1 越界
        b_i8 = tl.load(b_ptrs)
        # 动态反量化：INT8 → BF16，乘以 per-channel scale
        # 全在寄存器内完成，不走 HBM
        b    = b_i8.to(OUTPUT_DTYPE) * scale[None, :]   # [BLOCK_K, BLOCK_N] BF16
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(OUTPUT_DTYPE), mask=m_mask & (rn[None, :] < N))


def matmul_dequant(
        a: torch.Tensor,
        b_int8: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor = None,
) -> torch.Tensor:
    """
    a:      [M, K]  BF16/FP16
    b_int8: [K, N]  torch.int8
    scale:  [N]     BF16/FP16
    """
    assert a.ndim == 2 and b_int8.ndim == 2
    M, K = a.shape
    _, N = b_int8.shape

    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N) and out.device == a.device
        c = out

    output_dtype = _DTYPE_MAP.get(c.dtype, tl.bfloat16)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    matmul_dequant_kernel[grid](
        a, b_int8, scale, c,
        M, N, K,
        a.stride(0),      a.stride(1),
        b_int8.stride(0), b_int8.stride(1),
        c.stride(0),      c.stride(1),
        OUTPUT_DTYPE=output_dtype,
    )
    return c


def benchmark():
    WARMUP = 100
    REPEAT = 1000
    dtype  = torch.bfloat16
    device = "cuda"

    shapes = [
        (32, 11008, 4096,  "down proj [32,11008]x[11008,4096]"),
        (32, 4096,  12288, "QKV proj  [32,4096]x[4096,12288]"),
        (1,  11008, 4096,  "down proj [1,11008]x[11008,4096]"),
        (1,  4096,  12288, "QKV proj  [1,4096]x[4096,12288]"),
    ]

    print("=" * 110)
    print(f"{'Shape':<40} {'torch(bf16)':>12} {'v3(bf16)':>10} {'dequant(w8)':>12} {'vs torch':>10} {'误差(cosine)':>14}")
    print("=" * 110)

    total_torch = total_v3 = total_dq = 0.0

    for M, K, N, label in shapes:
        a    = torch.randn(M, K, dtype=dtype, device=device)
        b_fp = torch.randn(K, N, dtype=dtype, device=device)
        c_buf = torch.empty(M, N, dtype=dtype, device=device)

        # ── 量化：torch 原生 per-channel 对称量化 ────────────────────────
        scales      = b_fp.float().abs().max(dim=0).values.clamp(min=1e-5) / 127.0
        zero_points = torch.zeros(N, dtype=torch.int32, device=device)
        b_q    = torch.quantize_per_channel(
                     b_fp.float(), scales, zero_points, axis=1, dtype=torch.qint8)
        b_int8 = b_q.int_repr()                           # [K, N] torch.int8
        scale  = b_q.q_per_channel_scales().to(dtype)     # [N]

        # ── warmup ───────────────────────────────────────────────────────
        for _ in range(WARMUP):
            torch.matmul(a, b_fp, out=c_buf)
            matmul_v3(a, b_fp, out=c_buf)
            matmul_dequant(a, b_int8, scale, out=c_buf)
        torch.cuda.synchronize()

        # ── torch.matmul (BF16) ──────────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            torch.matmul(a, b_fp, out=c_buf)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / REPEAT * 1000
        ref = torch.matmul(a, b_fp)

        # ── matmul_v3 (BF16) ────────────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3(a, b_fp, out=c_buf)
        torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) / REPEAT * 1000

        # ── matmul_dequant (W8A16) ───────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_dequant(a, b_int8, scale, out=c_buf)
        torch.cuda.synchronize()
        t_dq = (time.perf_counter() - t0) / REPEAT * 1000

        # ── 精度：cosine similarity ──────────────────────────────────────
        out_dq = matmul_dequant(a, b_int8, scale)
        cos = torch.nn.functional.cosine_similarity(
            ref[:M].float(), out_dq[:M].float(), dim=1,
        ).mean().item()

        total_torch += t_torch
        total_v3    += t_v3
        total_dq    += t_dq

        print(f"{label:<40} {t_torch*1000:>9.1f}μs  {t_v3*1000:>9.1f}μs  {t_dq*1000:>9.1f}μs  "
              f"{t_torch/t_dq:>9.2f}x  {cos:>13.6f}")

    print("-" * 110)
    print(f"{'总延迟（4个shape累加）':<40} {total_torch*1000:>9.1f}μs  {total_v3*1000:>9.1f}μs  {total_dq*1000:>9.1f}μs")
    print(f"{'vs torch.matmul':<40} {'1.00x':>12} {total_torch/total_v3:>9.2f}x  {total_torch/total_dq:>11.2f}x")
    print("=" * 110)


if __name__ == "__main__":
    benchmark()
