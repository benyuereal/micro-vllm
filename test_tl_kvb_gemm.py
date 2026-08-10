#!/usr/bin/env python3
"""验证 TileLang T.gemm 能否在 L20 上做 [64,512]@[512,4096] (kvb 融合的关键) + bf16。
M=64 满足 M%16==0。这是 fused MLA kernel 内 kvb_proj 的形状。
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def kvb_gemm(N, K, OUT, BLOCK_N, BLOCK_K, BLOCK_OUT, dtype):
    accum = T.float32
    @T.prim_func
    def main(X: T.Tensor([N, K], dtype), W: T.Tensor([OUT, K], dtype), O: T.Tensor([N, OUT], dtype)):
        with T.Kernel(T.ceildiv(OUT, BLOCK_OUT), threads=256) as (bx):
            X_s = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            W_s = T.alloc_shared([BLOCK_OUT, BLOCK_K], dtype)
            acc = T.alloc_fragment([BLOCK_N, BLOCK_OUT], accum)
            T.fill(acc, 0)
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                T.copy(X[:, k*BLOCK_K:(k+1)*BLOCK_K], X_s)
                T.copy(W[bx*BLOCK_OUT:(bx+1)*BLOCK_OUT, k*BLOCK_K:(k+1)*BLOCK_K], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True)
            for i, j in T.Parallel(BLOCK_N, BLOCK_OUT):
                O[i, bx*BLOCK_OUT+j] = acc[i, j]
    return main


def main():
    N, K, OUT = 64, 512, 4096
    for dt in [torch.float16, torch.bfloat16]:
        tl_dt = T.float16 if dt == torch.float16 else T.bfloat16
        kernel = kvb_gemm(N, K, OUT, 64, 64, 64, tl_dt)
        x = torch.randn(N, K, dtype=dt, device="cuda") * 0.1
        w = torch.randn(OUT, K, dtype=dt, device="cuda") * 0.1
        o = kernel(x, w)
        ref = (x @ w.t()).to(dt)
        diff = (o.float() - ref.float()).abs().max().item()
        print(f"{dt}: max_diff={diff:.4f}  shape={tuple(o.shape)}  OK={diff < 0.5}")
        # bench
        for _ in range(20): kernel(x, w)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(300): kernel(x, w)
        e.record(); torch.cuda.synchronize()
        print(f"  latency: {s.elapsed_time(e)/300*1000:.1f} us")


if __name__ == "__main__":
    main()
