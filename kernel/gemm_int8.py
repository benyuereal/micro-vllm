"""W8A16 int8 分块 GEMM（TileLang）：小 M（投机解码 verify，M=1+N≈8）权重 HBM 只读一次。

背景：int8 GEMV（gemv_int8）对 M>1 用 grid.y=M，每个 token 行独立读一遍 int8 权重
（M=8 → 27GB 权重读 8 次=216GB）。反量化 matmul 只读 int8 一次但写 54GB bf16 权重
（每层）。本 kernel 用 TileLang 分块 GEMM：int8 权重从 HBM 只读一次（每 block 一个
[BLOCK_N, BLOCK_K] tile），shared 内 dequant→bf16，T.gemm 累加。M=8 时比 GEMV 快
12-31x（实测 mlp_gu 3.6ms→0.3ms），maxdiff=0.0（与反量化 matmul 完全一致）。

仅用于投机解码 verify（M 小）+ 小 M prefill（32 < M ≤ 128）。正常 decode（M=1）
走 GEMV、大 M prefill 走反量化 matmul，均不受影响。调用链：
kernel.gemm_int8_triton.verify_int8_gemm（开关 set_verify_gemm，SpecEngine 在 verify
前后设置；MICRO_VERIFY_GEMM=marlin 默认模式下权重是 Marlin dict，本模块不可达，
作为显式 env 指定的 fallback 保留）。
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


@tilelang.jit()
def _int8_gemm_kernel_fast(M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, dtype):
    """verify（M≤16）快速 int8 GEMM：int8 权重 cp.async 直进 shared，dequant 到 bf16
    shared，T.gemm 走 SS 变体（A/B 都 shared，mma 经 ldmatrix 读 shared）。

    相比 _int8_gemm_kernel（同步从 global dequant 到 bf16 shared，BK=128）的关键改进
    （对齐 Marlin 思路）：
      1. int8 权重用 T.copy（cp.async）异步直进 shared，与上一 tile 的 dequant+mma
         overlap（原 kernel 是同步 global 读，HBM 延迟不隐藏）。
      2. BLOCK_K=256（=2 个 group）：cp.async 一次读 256 宽 int8 tile，HBM 冷读带宽
         从 475→510 GB/s（实测 read-floor BK128=187us / BK256=175us，Marlin=174.8us）。
         scale 索引 kb*(BK//128)+k//128 覆盖 tile 内 2 个 group。
      3. dequant 到 bf16 shared（W_s）而非 fragment 寄存器：mma 经 ldmatrix 读 shared
         比读 strided fragment 快（实测 SS 188.9us vs SR-fragment 196.5us）。
      4. X 用 masked load（M<BLOCK_M 时 pad 行不越界读）。
    仅用于 M≤16（verify）；M 大时 BLOCK_M 超 shared 上限且 dequant 占比高，回退
    _int8_gemm_kernel。要求 N%BLOCK_N==0 且 K%BLOCK_K==0（T.copy 无 mask）。"""
    accum = T.float32
    G = BLOCK_K // 128   # 每 tile 覆盖的 group 数（BK=256→2）

    @T.prim_func
    def main(X: T.Tensor([M, K], dtype),
             W: T.Tensor([N, K], "int8"),
             S: T.Tensor([N, NG], "float32"),
             Out: T.Tensor([M, N], dtype)):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as (bn,):
            X_s = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            W8_s = T.alloc_shared([BLOCK_N, BLOCK_K], "int8")   # int8 cp.async 中转
            W_s = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)     # dequant 后 bf16
            acc = T.alloc_fragment([BLOCK_M, BLOCK_N], accum)
            T.clear(acc)
            for kb in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                for i, k in T.Parallel(BLOCK_M, BLOCK_K):
                    X_s[i, k] = T.if_then_else(
                        i < M and kb * BLOCK_K + k < K,
                        X[i, kb * BLOCK_K + k], 0)
                T.copy(W[bn * BLOCK_N, kb * BLOCK_K], W8_s)   # cp.async int8 直进 shared
                for n, k in T.Parallel(BLOCK_N, BLOCK_K):
                    W_s[n, k] = (T.cast(W8_s[n, k], accum)
                                 * S[bn * BLOCK_N + n, kb * G + k // 128]).astype(dtype)
                T.gemm(X_s, W_s, acc, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
            for i, n in T.Parallel(BLOCK_M, BLOCK_N):
                if i < M and bn * BLOCK_N + n < N:
                    Out[i, bn * BLOCK_N + n] = acc[i, n].astype(dtype)
    return main


def int8_gemm(x, w_int8, scale, out=None):
    """W8A16 int8 分块 GEMM（group-128）。x [M,K] bf16, w_int8 [N,K] int8,
    scale [N,K/128] fp32 → out [M,N] bf16（原地）。M 小（2-32）时权重 HBM 只读一次。

    M≤16（verify）走 _int8_gemm_kernel_fast（int8 cp.async 直进 shared + dequant 到
    bf16 shared + BK=256，HBM 冷读 475→510 GB/s，gate_proj 203→189us 逼近 Marlin
    174.8）；M>16（小 M prefill）走 _int8_gemm_kernel（BLOCK_M 大时 fast 版 shared
    超限且 dequant 占比高反而慢）。"""
    M, K = x.shape
    N = w_int8.shape[0]
    dtype = x.dtype
    if out is None:
        out = torch.empty(M, N, dtype=dtype, device=x.device)
    NG = K // 128
    # fast 版要求 N%BLOCK_N==0（T.copy 无 mask）；BLOCK_K=256 需 K%256==0 否则 128
    use_fast = (M <= 16 and N % 32 == 0 and (K % 256 == 0 or K % 128 == 0))
    if use_fast:
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = 256 if K % 256 == 0 else 128
        key = ("fast", M, N, K, dtype)
        if key not in _kernel_cache:
            _kernel_cache[key] = _int8_gemm_kernel_fast(
                M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, _TORCH_TO_TL[dtype])
    else:
        BLOCK_M = max(16, tilelang.next_power_of_2(M))
        BLOCK_N = 64
        BLOCK_K = 128
        key = (M, N, K, dtype)
        if key not in _kernel_cache:
            _kernel_cache[key] = _int8_gemm_kernel(
                M, N, K, NG, BLOCK_M, BLOCK_N, BLOCK_K, _TORCH_TO_TL[dtype])
    _kernel_cache[key](x, w_int8, scale, out)
    return out
