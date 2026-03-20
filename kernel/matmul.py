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
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_kernel_v3(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr = tl.float16,
):
    # M=32 硬编码
    BLOCK_M: tl.constexpr = 32

    pid_n = tl.program_id(axis=0)

    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(OUTPUT_DTYPE))


def matmul_v3(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"
        assert out.device == a.device
        c = out

    output_dtype = _DTYPE_MAP.get(c.dtype, tl.float16)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

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

    # down proj: [32, 11008] x [11008, 4096]
    # QKV proj:  [32, 4096]  x [4096, 12288]
    shapes = [
        (32, 11008, 4096,  "down proj [32,11008]x[11008,4096]"),
        (32, 4096,  12288, "QKV proj  [32,4096]x[4096,12288]"),
        (1,  11008, 4096,  "down proj [1,11008]x[11008,4096]"),
        (1,  4096,  12288, "QKV proj  [1,4096]x[4096,12288]"),
    ]

    print("=" * 70)
    print(f"{'Shape':<40} {'torch.matmul':>12} {'matmul_v3':>12} {'加速比':>8}")
    print("=" * 70)

    for M, K, N, label in shapes:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
        c_out = torch.empty(M, N, dtype=dtype, device=device)

        # warmup
        for _ in range(WARMUP):
            torch.matmul(a, b)
            matmul_v3(a, b, out=c_out)
        torch.cuda.synchronize()

        # torch.matmul
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            torch.matmul(a, b, out=c_out)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / REPEAT * 1000

        # matmul_v3
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3(a, b, out=c_out)
        torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) / REPEAT * 1000

        # 带宽利用率
        weight_bytes = K * N * 2
        bw_torch = weight_bytes / t_torch / 1e6  # GB/s
        bw_v3    = weight_bytes / t_v3    / 1e6

        print(f"{label:<40} {t_torch*1000:>9.1f}μs  {t_v3*1000:>9.1f}μs  {t_torch/t_v3:>7.2f}x")
        print(f"  带宽利用率: torch={bw_torch:.0f}GB/s ({bw_torch/15.5:.0f}%)  v3={bw_v3:.0f}GB/s ({bw_v3/15.5:.0f}%)")

    print("=" * 70)


if __name__ == "__main__":
    benchmark()
