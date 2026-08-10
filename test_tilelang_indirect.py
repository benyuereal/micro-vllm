#!/usr/bin/env python3
"""验证 TileLang 间接寻址 + 手写 GEMV（fragment reduce）。

grouped GEMV: out[n] = X[n] @ W[idx[n]].T,  W: [E, OUT, H]
- T.gemm 要求 M%16==0，GEMV (M=1) 不能用 → 用 fragment elementwise + reduce_sum
- 间接寻址: e = IDX[bn]; T.copy(W[e, ...], W_shared)
- persistent: grid=N, 每 block 一个 token
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def grouped_gemv_indirect(N, H, OUT, E, BLOCK_OUT, BLOCK_H, num_stages=2):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        X: T.Tensor([N, H], dtype),
        W: T.Tensor([E, OUT, H], dtype),
        IDX: T.Tensor([N], T.int32),
        Out: T.Tensor([N, OUT], dtype),
    ):
        with T.Kernel(N, T.ceildiv(OUT, BLOCK_OUT), threads=128) as (bn, bo):
            X_shared = T.alloc_shared([BLOCK_H], dtype)
            W_shared = T.alloc_shared([BLOCK_OUT, BLOCK_H], dtype)
            acc = T.alloc_fragment([BLOCK_OUT], accum_dtype)
            prod = T.alloc_fragment([BLOCK_OUT, BLOCK_H], accum_dtype)

            T.fill(acc, 0)
            e = IDX[bn]

            for kh in T.Pipelined(T.ceildiv(H, BLOCK_H), num_stages=num_stages):
                T.copy(X[bn, kh * BLOCK_H:(kh + 1) * BLOCK_H], X_shared)
                T.copy(
                    W[e, bo * BLOCK_OUT:(bo + 1) * BLOCK_OUT, kh * BLOCK_H:(kh + 1) * BLOCK_H],
                    W_shared,
                )
                # prod[i,j] = W[i,j] * X[j]; acc[i] = sum_j prod[i,j]
                for i, j in T.Parallel(BLOCK_OUT, BLOCK_H):
                    prod[i, j] = W_shared[i, j].astype(accum_dtype) * X_shared[j].astype(accum_dtype)
                T.reduce_sum(prod, acc, dim=1, clear=False)

            T.copy(acc, Out[bn, bo * BLOCK_OUT:(bo + 1) * BLOCK_OUT])

    return main


def ref(X, W, IDX):
    N = X.shape[0]
    OUT = W.shape[1]
    out = torch.empty(N, OUT, dtype=X.dtype, device=X.device)
    for n in range(N):
        out[n] = X[n] @ W[IDX[n]].T
    return out


def main():
    N, H, OUT, E = 8, 2048, 1408, 64
    BLOCK_OUT, BLOCK_H = 128, 128

    torch.manual_seed(0)
    X = torch.randn(N, H, device="cuda", dtype=torch.float16)
    W = torch.randn(E, OUT, H, device="cuda", dtype=torch.float16) * 0.02
    IDX = torch.randint(0, E, (N,), device="cuda", dtype=torch.int32)

    kernel = grouped_gemv_indirect(N, H, OUT, E, BLOCK_OUT, BLOCK_H)
    print("compiled OK")

    out = kernel(X, W, IDX)
    print("kernel ran, out shape:", out.shape, out.dtype)

    ref_out = ref(X, W, IDX.long())
    diff = (out.float() - ref_out.float()).abs()
    print(f"max diff: {diff.max().item():.4f}, mean diff: {diff.mean().item():.6f}")
    print(f"ref norm: {ref_out.float().norm().item():.2f}, out norm: {out.float().norm().item():.2f}")
    ok = diff.max().item() < 0.5
    print("✅ 间接寻址 + 手写 GEMV 正确" if ok else "❌ 错误")

    if ok:
        from tilelang.profiler import do_bench
        lat = do_bench(lambda: kernel(X, W, IDX), warmup=50)
        print(f"TileLang grouped_gemv: {lat*1000:.2f} us (N={N})")


if __name__ == "__main__":
    main()
