"""
Fused GEMM1 + SwiGLU kernel.

Computes: out[M, N] = silu(A @ B_gate) * (A @ B_up)
where B = [B_up | B_gate] with shape [K, 2*N].

Eliminates the intermediate gate_up buffer write/read compared to:
    gate_up = matmul_v3(x, gu_weight)   # writes [M, 2N] to HBM
    swiglu(gate_up, swiglu_out)                 # reads [M, 2N] from HBM
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_swiglu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program computes one [32, BLOCK_N] output tile.
    M=32 硬编码，与 matmul_kernel_v3 完全一致。

    For output tile at column pid_n:
      up_tile   = A @ B[:, pid_n*BLOCK_N : (pid_n+1)*BLOCK_N]
      gate_tile = A @ B[:, N + pid_n*BLOCK_N : N + (pid_n+1)*BLOCK_N]
      out_tile  = silu(gate_tile) * up_tile
    """
    pid_n = tl.program_id(axis=0)

    # M=32 硬编码，与 matmul_kernel_v3 完全一致
    rm = tl.arange(0, 32)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # A: [M, K]，M 维度固定不动
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak

    # B_up:   B[:, rn]     — first N columns
    b_up_ptrs   = b_ptr + rk[:, None] * stride_bk + rn[None, :]       * stride_bn
    # B_gate: B[:, N + rn] — second N columns
    b_gate_ptrs = b_ptr + rk[:, None] * stride_bk + (rn[None, :] + N) * stride_bn

    acc_up   = tl.zeros((32, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((32, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a      = tl.load(a_ptrs)
        b_up   = tl.load(b_up_ptrs)
        b_gate = tl.load(b_gate_ptrs)

        acc_up   += tl.dot(a, b_up)
        acc_gate += tl.dot(a, b_gate)

        a_ptrs      += BLOCK_K * stride_ak
        b_up_ptrs   += BLOCK_K * stride_bk
        b_gate_ptrs += BLOCK_K * stride_bk

    # SwiGLU: silu(gate) * up  (silu(x) = x * sigmoid(x))
    out = acc_gate * tl.sigmoid(acc_gate) * acc_up

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out.to(tl.bfloat16))


def matmul_swiglu(
    a: torch.Tensor,    # [M, K]  normed hidden states
    b: torch.Tensor,    # [K, 2N] gate_up weight
    out: torch.Tensor,  # [M, N]  pre-allocated swiglu_out buffer
) -> None:
    """
    Fused GEMM1 + SwiGLU, writes result into pre-allocated `out`.

    Replaces:
        gate_up[:bs] = matmul_v3(normed[:bs], mlp._gu)
        swiglu(gate_up[:bs], swiglu_out[:bs])
    With:
        matmul_swiglu_fused(normed[:bs], mlp._gu, swiglu_out[:bs])
    """
    M, K = a.shape
    N = b.shape[1] // 2

    # BLOCK_K=64 (v3 用 128)，因为两个 accumulator 寄存器占用翻倍
    BLOCK_N    = 64
    BLOCK_K    = 64
    num_warps  = 4
    num_stages = 3

    grid = (N // BLOCK_N,)

    _matmul_swiglu_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0),   a.stride(1),
        b.stride(0),   b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ============================================================================
# 对比测试 & 性能测试
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    import time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from kernel.matmul import matmul_v3
    from kernel.swiglu import swiglu_gemm

    torch.manual_seed(42)
    device = "cuda"
    dtype  = torch.bfloat16

    # 典型 Qwen-7B decode 场景
    M, K, N_half = 32, 4096, 4096   # B 形状 [K, 2*N_half]

    x    = torch.randn(M, K,          device=device, dtype=dtype)
    b_gu = torch.randn(K, 2 * N_half, device=device, dtype=dtype)

    # -----------------------------------------------------------------------
    # 1. 参考值：torch fp32 matmul + silu
    # -----------------------------------------------------------------------
    gate_up_ref = x.float() @ b_gu.float()
    up_ref   = gate_up_ref[:, :N_half]
    gate_ref = gate_up_ref[:, N_half:]
    ref = (gate_ref * torch.sigmoid(gate_ref) * up_ref).to(dtype)

    # -----------------------------------------------------------------------
    # 2. 原始两步路径：matmul_v3 → swiglu_gemm
    # -----------------------------------------------------------------------
    gate_up_buf     = torch.empty(M, 2 * N_half, device=device, dtype=dtype)
    swiglu_out_orig = torch.empty(M, N_half,     device=device, dtype=dtype)
    gate_up_buf.copy_(matmul_v3(x, b_gu))
    swiglu_gemm(gate_up_buf, swiglu_out_orig)

    # -----------------------------------------------------------------------
    # 3. 融合路径
    # -----------------------------------------------------------------------
    swiglu_out_fused = torch.empty(M, N_half, device=device, dtype=dtype)
    matmul_swiglu(x, b_gu, swiglu_out_fused)

    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # 正确性对比
    # -----------------------------------------------------------------------
    def stats(name, t, ref):
        diff = (t.float() - ref.float()).abs()
        ok = torch.allclose(t.float(), ref.float(), rtol=1e-2, atol=1e-2)
        print(f"  {name:<25}  max={diff.max().item():.5f}  mean={diff.mean().item():.6f}  allclose={ok}")

    print("=" * 65)
    print(f"正确性验证  (M={M}, K={K}, N={N_half}, dtype={dtype})")
    print("=" * 65)
    stats("orig (v3 + swiglu_gemm)", swiglu_out_orig,  ref)
    stats("fused",                   swiglu_out_fused, ref)
    diff_pair = (swiglu_out_orig.float() - swiglu_out_fused.float()).abs()
    print(f"  {'orig vs fused':<25}  max={diff_pair.max().item():.5f}  mean={diff_pair.mean().item():.6f}")

    # -----------------------------------------------------------------------
    # 性能测试
    # -----------------------------------------------------------------------
    WARMUP = 50
    REPEAT = 500

    def bench_orig():
        gate_up_buf.copy_(matmul_v3(x, b_gu))
        swiglu_gemm(gate_up_buf, swiglu_out_orig)

    def bench_fused():
        matmul_swiglu(x, b_gu, swiglu_out_fused)

    for _ in range(WARMUP):
        bench_orig()
        bench_fused()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEAT):
        bench_orig()
    torch.cuda.synchronize()
    orig_ms = (time.perf_counter() - t0) / REPEAT * 1000

    t0 = time.perf_counter()
    for _ in range(REPEAT):
        bench_fused()
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / REPEAT * 1000

    print()
    print("=" * 65)
    print(f"性能测试  (M={M}, K={K}, N={N_half}, warmup={WARMUP}, repeat={REPEAT})")
    print("=" * 65)
    print(f"  matmul_v3 + swiglu_gemm : {orig_ms:.4f} ms")
    print(f"  matmul_swiglu_fused            : {fused_ms:.4f} ms")
    print(f"  加速比                          : {orig_ms / fused_ms:.3f}x")
    print("=" * 65)
