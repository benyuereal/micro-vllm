#!/usr/bin/env python3
"""对比 TileLang GEMV 不同写法的性能：fragment reduce vs T.gemm(M=16 pad)。

决定 MoE kernel 用哪种 GEMV 写法。
"""
import torch
import tilelang
import tilelang.language as T


# 写法1: fragment reduce（当前 tilelang_moe 用的）
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def gemv_fragment(H, OUT, BLOCK_H, BLOCK_OUT):
    dtype = T.float16
    ad = T.float32
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


# 写法2: T.gemm M=16 pad（tensor core）
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def gemv_gemm_pad16(H, OUT, BLOCK_H, BLOCK_OUT):
    dtype = T.float16
    ad = T.float32
    @T.prim_func
    def main(X: T.Tensor([16, H], dtype), W: T.Tensor([OUT, H], dtype), O: T.Tensor([16, OUT], dtype)):
        with T.Kernel(T.ceildiv(OUT, BLOCK_OUT), threads=128) as (bo,):
            Xs = T.alloc_shared([16, BLOCK_H], dtype)
            Ws = T.alloc_shared([BLOCK_OUT, BLOCK_H], dtype)
            acc = T.alloc_fragment([16, BLOCK_OUT], ad)
            T.fill(acc, 0)
            for kh in T.Pipelined(T.ceildiv(H, BLOCK_H), num_stages=2):
                T.copy(X[0:16, kh*BLOCK_H:(kh+1)*BLOCK_H], Xs)
                T.copy(W[bo*BLOCK_OUT:(bo+1)*BLOCK_OUT, kh*BLOCK_H:(kh+1)*BLOCK_H], Ws)
                T.gemm(Xs, Ws, acc, transpose_B=True)
            T.copy(acc, O[0:16, bo*BLOCK_OUT:(bo+1)*BLOCK_OUT])
    return main


# 写法3: T.gemm M=1 但用 1×H view（看能否绕过 M%16）
# 已知会失败，跳过


def bench(fn, warmup=50, iters=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000


def main():
    H, OUT = 2048, 2816  # 2*inter
    dev = "cuda"
    X1 = torch.randn(1, H, device=dev, dtype=torch.float16)
    X16 = torch.randn(16, H, device=dev, dtype=torch.float16)
    W = torch.randn(OUT, H, device=dev, dtype=torch.float16) * 0.02

    print("compiling fragment...")
    kf = gemv_fragment(H, OUT, 128, 128)
    print("compiling gemm_pad16...")
    kg = gemv_gemm_pad16(H, OUT, 128, 128)

    # 正确性
    of = kf(X1, W)
    og = kg(X16, W)
    ref = X1 @ W.T
    print(f"fragment: max diff {(of.float()-ref.float()).abs().max().item():.5f}")
    print(f"gemm_pad16: max diff {(og[0].float()-ref.float()).abs().max().item():.5f}")

    tf = bench(lambda: kf(X1, W))
    tg = bench(lambda: kg(X16, W))
    print(f"\nfragment GEMV [1,H]@[H,2816]: {tf:.2f} us")
    print(f"gemm_pad16  [16,H]@[H,2816]: {tg:.2f} us (单 token 有效, 15/16 浪费)")
    print(f"×6 expert (gate_up): fragment={tf*6:.1f}us  gemm_pad16={tg*6:.1f}us")
    print(f"×8 token ×6 expert:  fragment={tf*6*8:.1f}us  gemm_pad16={tg*6*8:.1f}us")


if __name__ == "__main__":
    main()
