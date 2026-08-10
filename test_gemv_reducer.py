#!/usr/bin/env python3
"""测试 TileLang gemv_alloc_reducer 模式的 GEMV 性能，对比 fragment 写法。"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


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


# 间接寻址版（MoE 用）：a 是 [E, M, N]，用 idx 索引第 0 维
@tilelang.jit(out_idx=[3], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def gemv_reducer_indirect(E, M, N, K_tok, block_M, block_N, num_stages, threads,
                          dtype=T.float16, accum_dtype=T.float32):
    """a: [E, M, N], x: [K_tok, N], idx: [K_tok] -> o: [K_tok, M]
    每 block 一个 token，间接索引 expert 权重。"""
    @T.prim_func
    def main(a: T.Tensor((E, M, N), dtype), x: T.Tensor((K_tok, N), dtype),
             idx: T.Tensor((K_tok,), T.int32),
             o: T.Tensor((K_tok, M), dtype)):
        with T.Kernel(K_tok, T.ceildiv(M, block_M), threads=threads) as (kt, i0_m):
            o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(o_reducer)
            e = idx[kt]
            for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                a_smem = T.alloc_shared((block_M, block_N), dtype)
                T.copy(a[e, i0_m * block_M, i0_n * block_N], a_smem)
                a_frag = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(a_smem, a_frag)
                x_frag = T.alloc_fragment(block_N, dtype)
                T.copy(x[kt, i0_n * block_N], x_frag)
                for i1_m, i1_n in T.Parallel(block_M, block_N):
                    o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]
            T.finalize_reducer(o_reducer)
            T.copy(o_reducer, o[kt, i0_m * block_M])
    return main


def main():
    M, N = 2816, 2048
    dev = "cuda"
    a = torch.randn(M, N, device=dev, dtype=torch.float16)
    x = torch.randn(N, device=dev, dtype=torch.float16)

    k = gemv_reducer(M, N, 128, 128, 2, 256)
    o = k(a, x)
    ref = a @ x
    print(f"reducer diff: {(o.float()-ref.float()).abs().max().item():.5f}")
    t = do_bench(lambda: k(a, x), warmup=50)
    print(f"gemv_reducer [2816,2048]: {t*1000:.2f} us")

    # 不同 block 配置
    for bm, bn, ns, th in [(128,128,2,256),(64,128,2,128),(128,64,2,256),(256,64,2,256),(64,64,2,128)]:
        try:
            kk = gemv_reducer(M, N, bm, bn, ns, th)
            tt = do_bench(lambda: kk(a, x), warmup=30)
            print(f"  bm={bm} bn={bn} ns={ns} th={th}: {tt*1000:.2f} us")
        except Exception as e:
            print(f"  bm={bm} bn={bn} ns={ns} th={th}: FAIL {str(e)[:50]}")

    # 间接寻址版（MoE 实际场景：K_tok=8 token, E=64 expert）
    E, K_tok = 64, 8
    a3 = torch.randn(E, M, N, device=dev, dtype=torch.float16) * 0.02
    x2 = torch.randn(K_tok, N, device=dev, dtype=torch.float16)
    idx = torch.randint(0, E, (K_tok,), device=dev, dtype=torch.int32)
    ki = gemv_reducer_indirect(E, M, N, K_tok, 128, 128, 2, 256)
    oi = ki(a3, x2, idx)
    ref_i = torch.stack([x2[t] @ a3[idx[t]].T for t in range(K_tok)])
    print(f"\nindirect diff: {(oi.float()-ref_i.float()).abs().max().item():.5f}")
    ti = do_bench(lambda: ki(a3, x2, idx), warmup=50)
    print(f"gemv_reducer_indirect (8 token, 2816 out): {ti*1000:.2f} us")
    print(f"  ×6 expert (gate_up): {ti*6:.1f} us")


if __name__ == "__main__":
    main()
