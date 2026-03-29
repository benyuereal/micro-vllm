import torch
import triton
import triton.language as tl
import time


_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.autotune(
    configs=[
        # M=32 原有场景（down proj / QKV proj）
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 8}),
        # lm_head 大 N 场景
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 8}),
        # 小 batch（lm_head bs < 32）
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_v3(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr = tl.float16,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N

    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K
        mask_k = (k_off + rk) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(OUTPUT_DTYPE), mask=mask_m[:, None] & mask_n[None, :])


def matmul_v3(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"
        assert out.device == a.device
        c = out

    output_dtype = _DTYPE_MAP.get(c.dtype, tl.float16)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_kernel_v3[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        OUTPUT_DTYPE=output_dtype,
    )
    return c


def benchmark():
    import torch
    WARMUP = 100
    REPEAT = 1000
    dtype = torch.bfloat16
    device = "cuda"

    # M, K, N, label
    shapes = [
        # 原有 decode GEMM（M=32）
        (32, 11008, 4096,   "down proj  [32,11008]x[11008,4096]"),
        (32, 4096,  12288,  "QKV proj   [32,4096]x[4096,12288]"),
        (1,  11008, 4096,   "down proj  [1,11008]x[11008,4096]"),
        (1,  4096,  12288,  "QKV proj   [1,4096]x[4096,12288]"),
        # lm_head：Qwen2-7B vocab=151936
        (1,  4096,  151936, "lm_head    [1,4096]x[4096,151936]"),
        (8,  4096,  151936, "lm_head    [8,4096]x[4096,151936]"),
        (16, 4096,  151936, "lm_head    [16,4096]x[4096,151936]"),
        (32, 4096,  151936, "lm_head    [32,4096]x[4096,151936]"),
        (40, 4096,  151936, "lm_head    [40,4096]x[4096,151936]"),
    ]

    a100_bw = 1555  # GB/s

    print("=" * 80)
    print(f"{'Shape':<42} {'torch(μs)':>9} {'v3(μs)':>9} {'加速':>7} {'带宽%(v3)':>10}")
    print("=" * 80)

    for M, K, N, label in shapes:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
        c_out = torch.empty(M, N, dtype=dtype, device=device)

        for _ in range(WARMUP):
            torch.matmul(a, b)
            matmul_v3(a, b, out=c_out)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            torch.matmul(a, b, out=c_out)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / REPEAT * 1e6  # μs

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3(a, b, out=c_out)
        torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) / REPEAT * 1e6  # μs

        # 带宽：主要瓶颈是权重矩阵 B 的读取
        weight_bytes = K * N * 2  # BF16
        bw_v3 = weight_bytes / (t_v3 / 1e6) / 1e9  # GB/s

        print(f"{label:<42} {t_torch:>9.1f} {t_v3:>9.1f} {t_torch/t_v3:>7.2f}x {bw_v3/a100_bw*100:>9.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    benchmark()
