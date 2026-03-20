import torch
import triton
import triton.language as tl


_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


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
    pid = tl.program_id(axis=0)
    pid_n = pid

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

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 128
    num_warps = 4
    num_stages = 4

    grid = (N // BLOCK_N,)

    matmul_kernel_v3[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
        OUTPUT_DTYPE=output_dtype,
    )
    return c


def benchmark():
    import time
    from matmul import matmul_v3 as matmul_v3_new

    WARMUP = 100
    REPEAT = 1000
    dtype = torch.bfloat16
    device = "cuda"

    shapes = [
        (32, 11008, 4096,  "down proj [32,11008]x[11008,4096]"),
        (32, 4096,  12288, "QKV proj  [32,4096]x[4096,12288]"),
        (1,  11008, 4096,  "down proj [1,11008]x[11008,4096]"),
        (1,  4096,  12288, "QKV proj  [1,4096]x[4096,12288]"),
    ]

    print("=" * 85)
    print(f"{'Shape':<40} {'torch.matmul':>12} {'旧matmul_v3':>12} {'新matmul_v3':>12}")
    print("=" * 85)

    total_torch = total_old = total_new = 0.0

    for M, K, N, label in shapes:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
        c_buf = torch.empty(M, N, dtype=dtype, device=device)

        for _ in range(WARMUP):
            torch.matmul(a, b, out=c_buf)
            matmul_v3(a, b, out=c_buf)
            matmul_v3_new(a, b, out=c_buf)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            torch.matmul(a, b, out=c_buf)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / REPEAT * 1000

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3(a, b, out=c_buf)
        torch.cuda.synchronize()
        t_old = (time.perf_counter() - t0) / REPEAT * 1000

        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3_new(a, b, out=c_buf)
        torch.cuda.synchronize()
        t_new = (time.perf_counter() - t0) / REPEAT * 1000

        total_torch += t_torch
        total_old   += t_old
        total_new   += t_new

        print(f"{label:<40} {t_torch*1000:>9.1f}μs  {t_old*1000:>9.1f}μs  {t_new*1000:>9.1f}μs")

    print("-" * 85)
    print(f"{'总延迟（4个shape累加）':<40} {total_torch*1000:>9.1f}μs  {total_old*1000:>9.1f}μs  {total_new*1000:>9.1f}μs")
    print(f"{'vs torch.matmul':<40} {'1.00x':>12} {total_torch/total_old:>11.2f}x  {total_torch/total_new:>11.2f}x")
    print("=" * 85)


if __name__ == "__main__":
    benchmark()
