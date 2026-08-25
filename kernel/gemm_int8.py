"""W8A16 int8 分块 GEMM（TileLang）：小 M（投机解码 verify，M=1+N≈8）权重 HBM 只读一次。

背景：int8 GEMV（gemv_int8）对 M>1 用 grid.y=M，每个 token 行独立读一遍 int8 权重
（M=8 → 27GB 权重读 8 次=216GB）。反量化 matmul 只读 int8 一次但写 54GB bf16 权重
（每层）。本 kernel 用 TileLang 分块 GEMM：int8 权重从 HBM 只读一次（每 block 一个
[BLOCK_N, BLOCK_K] tile），shared 内 dequant→bf16，T.gemm 累加。M=8 时比 GEMV 快
12-31x（实测 mlp_gu 3.6ms→0.3ms），maxdiff=0.0（与反量化 matmul 完全一致）。

仅用于投机解码 verify（M 小）。正常 decode（M=1）走 GEMV、prefill（M 大）走反量化
matmul，均不受影响（由 kernel.gemm_int8_triton.set_verify_gemm 开关控制，
MICRO_VERIFY_GEMM=tilelang|triton 选后端，本模块是 tilelang 后端）。
"""
import torch
import tilelang
import tilelang.language as T

# (M, N, K, dtype) → 编译好的 kernel
_kernel_cache = {}

_TORCH_TO_TL = {
    torch.bfloat16: T.bfloat16,
    torch.float16: T.float16,
}


@tilelang.jit()
def _int8_gemm_kernel(M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, dtype):
    """out[M,N] bf16 = x[M,K] bf16 @ (w_int8[N,K] dequant group-128)^T。
    scale[N, NG] fp32，NG=K//128。BLOCK_K=128（=group，tile 内 scale 每行恒定）。
    权重 int8 从 HBM 只读一次（每 block 一个 [BLOCK_N, BLOCK_K] tile，block 间 N 不重叠）。
    out 由调用方提供（out_idx=[]），原地写入（对齐 adapter._lin 的 out 语义）。"""
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
                for i, k in T.Parallel(BLOCK_M, BLOCK_K):
                    X_s[i, k] = T.if_then_else(
                        i < M and kb * BLOCK_K + k < K,
                        X[i, kb * BLOCK_K + k], 0)
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


def int8_gemm_available() -> bool:
    return True


def int8_gemm(x, w_int8, scale, out=None):
    """W8A16 int8 分块 GEMM（group-128）。x [M,K] bf16, w_int8 [N,K] int8,
    scale [N,K/128] fp32 → out [M,N] bf16（原地）。M 小（2-32）时权重 HBM 只读一次。"""
    M, K = x.shape
    N = w_int8.shape[0]
    dtype = x.dtype
    if out is None:
        out = torch.empty(M, N, dtype=dtype, device=x.device)
    NG = K // 128
    BLOCK_M = max(16, tilelang.next_power_of_2(M))
    BLOCK_N = 64
    BLOCK_K = 128
    key = (M, N, K, dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = _int8_gemm_kernel(
            M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, _TORCH_TO_TL[dtype])
    _kernel_cache[key](x, w_int8, scale, out)
    return out
