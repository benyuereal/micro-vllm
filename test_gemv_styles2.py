#!/usr/bin/env python3
"""对比更多 GEMV 写法：shared reduce, 不同 tile, 看能否逼近 Triton 12us/单token-expert。"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


# 写法A: shared 上 W*X 写 shared, reduce_sum shared->frag
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def gemv_shared_reduce(H, OUT, BH, BO):
    dtype = T.float16; ad = T.float32
    @T.prim_func
    def main(X: T.Tensor([1, H], dtype), W: T.Tensor([OUT, H], dtype), O: T.Tensor([1, OUT], dtype)):
        with T.Kernel(T.ceildiv(OUT, BO), threads=256) as (bo,):
            Xs = T.alloc_shared([BH], dtype)
            Ws = T.alloc_shared([BO, BH], dtype)
            prod_s = T.alloc_shared([BO, BH], ad)
            acc = T.alloc_fragment([BO], ad)
            T.fill(acc, 0)
            for kh in T.Pipelined(T.ceildiv(H, BH), num_stages=2):
                T.copy(X[0, kh*BH:(kh+1)*BH], Xs)
                T.copy(W[bo*BO:(bo+1)*BO, kh*BH:(kh+1)*BH], Ws)
                for i, j in T.Parallel(BO, BH):
                    prod_s[i, j] = Ws[i, j].astype(ad) * Xs[j].astype(ad)
                T.reduce_sum(prod_s, acc, dim=1, clear=False)
            T.copy(acc, O[0, bo*BO:(bo+1)*BO])
    return main


# 写法B: fragment 但 W load 到 shared, X broadcast, 用更大的 BO
@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def gemv_frag_bigout(H, OUT, BH, BO):
    dtype = T.float16; ad = T.float32
    @T.prim_func
    def main(X: T.Tensor([1, H], dtype), W: T.Tensor([OUT, H], dtype), O: T.Tensor([1, OUT], dtype)):
        with T.Kernel(T.ceildiv(OUT, BO), threads=256) as (bo,):
            Xs = T.alloc_shared([BH], dtype)
            Ws = T.alloc_shared([BO, BH], dtype)
            acc = T.alloc_fragment([BO], ad)
            prod = T.alloc_fragment([BO, BH], ad)
            T.fill(acc, 0)
            for kh in T.Pipelined(T.ceildiv(H, BH), num_stages=2):
                T.copy(X[0, kh*BH:(kh+1)*BH], Xs)
                T.copy(W[bo*BO:(bo+1)*BO, kh*BH:(kh+1)*BH], Ws)
                for i, j in T.Parallel(BO, BH):
                    prod[i, j] = Ws[i, j].astype(ad) * Xs[j].astype(ad)
                T.reduce_sum(prod, acc, dim=1, clear=False)
            T.copy(acc, O[0, bo*BO:(bo+1)*BO])
    return main


def main():
    H, OUT = 2048, 2816
    dev = "cuda"
    X = torch.randn(1, H, device=dev, dtype=torch.float16)
    W = torch.randn(OUT, H, device=dev, dtype=torch.float16) * 0.02
    ref = X @ W.T

    for name, fn, cfgs in [
        ("shared_reduce", gemv_shared_reduce, [(128,128),(256,128),(128,256)]),
        ("frag_bigout", gemv_frag_bigout, [(128,256),(256,256),(128,128)]),
    ]:
        for BH, BO in cfgs:
            try:
                k = fn(H, OUT, BH, BO)
                o = k(X, W)
                d = (o.float()-ref.float()).abs().max().item()
                t = do_bench(lambda: k(X, W), warmup=50)
                print(f"{name} BH={BH} BO={BO}: {t*1000:.2f} us  diff={d:.5f}")
            except Exception as e:
                print(f"{name} BH={BH} BO={BO}: FAIL {str(e)[:50]}")


if __name__ == "__main__":
    main()
