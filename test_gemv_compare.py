#!/usr/bin/env python3
"""统一 CUDA event 手动计时对比：Triton GEMV vs TileLang fragment vs TileLang reducer。

用相同测量方法，避免 do_bench backend 差异。
"""
import torch
import tilelang
import tilelang.language as T
import sys
sys.path.insert(0, "/models/micro-vllm")
from kernel.grouped_gemv import grouped_gate_up, grouped_down


def bench_event(fn, warmup=50, iters=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


# TileLang fragment GEMV（之前的写法）
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def gemv_fragment(H, OUT, BLOCK_H, BLOCK_OUT):
    dtype = T.float16; ad = T.float32
    @T.prim_func
    def main(X: T.Tensor([1, H], dtype), W: T.Tensor([OUT, H], dtype), O: T.Tensor([1, OUT], dtype)):
        with T.Kernel(T.ceildiv(OUT, BLOCK_OUT), threads=128) as (bo,):
            Xs = T.alloc_shared([BLOCK_H], dtype)
            Ws = T.alloc_shared([BLOCK_OUT, BLOCK_H], dtype)
            acc = T.alloc_fragment([BLOCK_OUT], ad)
            prod = T.alloc_fragment([BLOCK_OUT, BLOCK_H], ad)
            T.fill(acc, 0)
            for kh in T.Pipelined(T.ceildiv(H, BLOCK_H), num_stages=2):
                T.copy(X[0, kh*BLOCK_H:(kh+1)*BLOCK_H], Xs)
                T.copy(W[bo*BLOCK_OUT:(bo+1)*BLOCK_OUT, kh*BLOCK_H:(kh+1)*BLOCK_H], Ws)
                for i, j in T.Parallel(BLOCK_OUT, BLOCK_H):
                    prod[i, j] = Ws[i, j].astype(ad) * Xs[j].astype(ad)
                T.reduce_sum(prod, acc, dim=1, clear=False)
            T.copy(acc, O[0, bo*BLOCK_OUT:(bo+1)*BLOCK_OUT])
    return main


@tilelang.jit(out_idx=[2], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def gemv_reducer(M, N, block_M, block_N, num_stages, threads,
                 dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(a: T.Tensor((M, N), dtype), x: T.Tensor((N,), dtype),
             o: T.Tensor((M,), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as i0_m:
            o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(o_reducer)
            for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                a_smem = T.alloc_shared((block_M, block_N), dtype)
                T.copy(a[i0_m * block_M, i0_n * block_N], a_smem)
                a_frag = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(a_smem, a_frag)
                x_frag = T.alloc_fragment(block_N, dtype)
                T.copy(x[i0_n * block_N], x_frag)
                for i1_m, i1_n in T.Parallel(block_M, block_N):
                    o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]
            T.finalize_reducer(o_reducer)
            T.copy(o_reducer, o[i0_m * block_M])
    return main


def main():
    H, OUT = 2048, 2816
    E, K = 64, 6
    dev = "cuda"
    dt = torch.float16

    # Triton grouped_gate_up: x[1,H] @ e_gu[idx, OUT, H].T, K experts
    x1 = torch.randn(1, H, device=dev, dtype=dt)
    e_gu = torch.randn(E, OUT, H, device=dev, dtype=dt) * 0.02
    idx = torch.randint(0, E, (K,), device=dev, dtype=torch.int64)

    def triton_gate_up():
        return grouped_gate_up(x1, e_gu, idx)  # [K, OUT]

    # TileLang fragment: 单 token 单 expert [1,H]@[H,OUT]
    x_tl = torch.randn(1, H, device=dev, dtype=dt)
    W_tl = torch.randn(OUT, H, device=dev, dtype=dt) * 0.02
    kf = gemv_fragment(H, OUT, 128, 128)
    def tl_fragment():
        return kf(x_tl, W_tl)  # [1, OUT]

    # TileLang reducer
    a_r = torch.randn(OUT, H, device=dev, dtype=dt) * 0.02
    x_r = torch.randn(H, device=dev, dtype=dt)
    kr = gemv_reducer(OUT, H, 128, 128, 2, 256)
    def tl_reducer():
        return kr(a_r, x_r)  # [OUT]

    # 单 expert GEMV [1,H]@[H,2816]
    print("=== 单 expert GEMV [1,2048]@[2048,2816] ===")
    print(f"  Triton grouped_gate_up (K=6 expert): {bench_event(triton_gate_up):.2f} us  (6 expert 一起)")
    print(f"  TileLang fragment (1 expert):        {bench_event(tl_fragment):.2f} us")
    print(f"  TileLang reducer  (1 expert):        {bench_event(tl_reducer):.2f} us")
    print(f"  → fragment ×6 = {bench_event(tl_fragment)*6:.1f} us, reducer ×6 = {bench_event(tl_reducer)*6:.1f} us")

    # N=8 token 完整 MoE（gate_up + down）
    N = 8
    x8 = torch.randn(N, H, device=dev, dtype=dt)
    idx8 = torch.randint(0, E, (N, K), device=dev, dtype=torch.int64)
    w8 = torch.rand(N, K, device=dev, dtype=dt)
    e_d = torch.randn(E, H, 1408, device=dev, dtype=dt) * 0.02
    w_ones = torch.ones(K, device=dev, dtype=dt)
    out8 = torch.empty(N, H, device=dev, dtype=dt)

    def triton_full():
        for i in range(N):
            ii = idx8[i]
            gu = grouped_gate_up(x8[i:i+1], e_gu, ii)
            g, u = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(g) * u * w8[i].unsqueeze(-1).to(gu.dtype)
            out8[i:i+1] = grouped_down(act, e_d, ii, w_ones)

    print(f"\n=== N=8 token 完整 routed SwiGLU (16 kernel) ===")
    print(f"  Triton loop: {bench_event(triton_full):.2f} us")


if __name__ == "__main__":
    main()
