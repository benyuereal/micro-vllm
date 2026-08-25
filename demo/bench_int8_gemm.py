"""验证 TileLang int8 GEMM（W8A16，group-128）：M=8 时权重只读一次 vs GEMV 读 M 次。

对比：
  1. 现有 w8_linear（int8 GEMV，M=8 → 权重读 8 次）
  2. TileLang 分块 GEMM（int8 权重 HBM 只读一次，shared 内 dequant→bf16，T.gemm）
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import tilelang
import tilelang.language as T

from kernel.gemv_int8 import w8_linear, gemv_int8_available


@tilelang.jit()
def int8_gemm_kernel(M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, dtype):
    """out[M,N] bf16 = x[M,K] bf16 @ (w_int8[N,K] dequant group-128)^T。
    scale[N, NG] fp32，NG=K//128。BLOCK_K=128（=group，tile 内 scale 每行恒定）。
    权重 int8 从 HBM 只读一次（每 block 一个 [BLOCK_N, BLOCK_K] tile）。"""
    accum = T.float32

    @T.prim_func
    def main(X: T.Tensor([M, K], dtype),
             W: T.Tensor([N, K], "int8"),
             S: T.Tensor([N, NG], "float32"),
             Out: T.Tensor([M, N], dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as (bn,):
            X_s = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            W_s = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)   # dequant 后 bf16
            acc = T.alloc_fragment([BLOCK_M, BLOCK_N], accum)
            T.clear(acc)
            for kb in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                # 载入 x tile（M 小，pad 到 BLOCK_M）
                for i, k in T.Parallel(BLOCK_M, BLOCK_K):
                    X_s[i, k] = T.if_then_else(
                        i < M and kb * BLOCK_K + k < K,
                        X[i, kb * BLOCK_K + k], 0)
                # 载入 w int8 tile → dequant 成 bf16（scale 每行恒定，group=BLOCK_K）
                for n, k in T.Parallel(BLOCK_N, BLOCK_K):
                    wv = T.cast(W[bn * BLOCK_N + n, kb * BLOCK_K + k], accum)
                    sv = S[bn * BLOCK_N + n, kb]
                    W_s[n, k] = wv * sv
                T.gemm(X_s, W_s, acc, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
            for i, n in T.Parallel(BLOCK_M, BLOCK_N):
                if i < M and bn * BLOCK_N + n < N:
                    Out[i, bn * BLOCK_N + n] = acc[i, n].astype(dtype)
    return main


def make_int8_gemm(M, N, K, dtype, device):
    NG = K // 128
    BLOCK_M = max(16, tilelang.next_power_of_2(M))
    BLOCK_N = 64
    BLOCK_K = 128
    fn = int8_gemm_kernel(M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, dtype)

    def run(x, w_int8, scale, out=None):
        # out_idx=[]：out 由调用方提供，kernel 原地写入（对齐 adapter._lin 的 out 语义）
        if out is None:
            out = torch.empty(M, N, dtype=dtype, device=device)
        fn(x, w_int8, scale, out)
        return out
    return run


def bench(fn, iters=20):
    torch.cuda.synchronize()
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"gemv_int8_available={gemv_int8_available()}")
    # 代表性层：MLP gate_up N=34816 K=5120；GDN qz N=16384 K=5120；full qkv N=13312 K=5120
    for name, N, K in [("mlp_gu", 34816, 5120), ("gdn_qz", 16384, 5120),
                       ("full_qkv", 13312, 5120), ("mlp_down", 5120, 17408)]:
        for M in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(M, K, dtype=dtype, device=device)
            w_int8 = (torch.randn(N, K, device=device) * 10).to(torch.int8)
            scale = torch.rand(N, K // 128, dtype=torch.float32, device=device) + 0.5
            out_g = torch.empty(M, N, dtype=dtype, device=device)
            t_g = bench(lambda: w8_linear(x, w_int8, scale, out_g, "MICRO_GEMV"))
            # TileLang int8 GEMM
            try:
                run = make_int8_gemm(M, N, K, dtype, device)
                out_t = run(x, w_int8, scale)
                # 正确性：与反量化 matmul 对比
                sc = scale.repeat_interleave(128, dim=1)
                ref = torch.matmul(x, (w_int8.float() * sc).to(dtype).t())
                diff = (out_t.float() - ref.float()).abs().max().item()
                t_t = bench(lambda: run(x, w_int8, scale))
                print(f"{name} M={M}: GEMV={t_g:.1f}ms  TileLang={t_t:.1f}ms "
                      f"speedup={t_g/t_t:.2f}x  maxdiff={diff:.4f}")
            except Exception as e:
                print(f"{name} M={M}: GEMV={t_g:.1f}ms  TileLang FAILED: {e}")


if __name__ == "__main__":
    main()
