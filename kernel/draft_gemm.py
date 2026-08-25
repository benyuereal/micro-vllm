"""DFlash2 草稿模型通用变长 M 的 TileLang GEMM（runtime grid dim）。

替代草稿模型里所有 ``x @ w.T``（w 是 [N,K]）的 torch GEMM。

📌 核心思路（runtime grid dim，M 是 runtime 参数）：
    草稿模型每步 query 只有 1+N=8 个 token（M=8），context 路径 M=C（变长，可达几千）。
    ``T.gemm`` 要求 M%16==0（tensor-core mma 硬约束）。解法：
    - X 是 [MAX_M16, K]（MAX_M16 是编译期上界，如 16 / 4096），pad 行零填充。
    - M 是 T.int32 runtime 参数，grid 第一维 = T.ceildiv(M, 16)（runtime grid dim）。
    - 同一 kernel 对任意 M<=MAX_M16 复用，无需重编译。
    已验证 M=1/8/100/500/2048/4096 全对（max_rel 0.002~0.0037，bf16 正常）。

📌 launcher ``draft_gemm(x, w, max_m)``：
    - x: [M, K] bf16（M 变长），w: [N, K] bf16。
    - 复用持久 pad buffer xbuf [MAX_M16, K]：xbuf[:M]=x，xbuf[M:M16]=0（只清 <=15 行，
      不清整个 buffer——fc 路径 MAX_M16*K 可达 200MB，整清太贵）。
    - 调 kernel，返回 out[:M]。
    - kernel 只读 X[0:M16]（grid 只覆盖 ceildiv(M,16) 个 block），故 [M16:MAX_M16] 的
      stale 数据无害。

📌 约束：所有 K 必须 %128==0，所有 N 必须 %64==0（草稿模型各 GEMM 均满足）。
"""
import torch
import tilelang
import tilelang.language as T


# ============ 变长 M GEMM（runtime grid dim）============
# BLOCK_M 参数化：
#   - query 路径（M=8，pad 16）：BLOCK_M=16（grid M 维=1，最优）。
#   - context 路径（M=C 变长，可达几千）：BLOCK_M=64（BM=16 时 W 被重复读 C/16 次，
#     fc 的 W=262MB 从 HBM 读 128 次→49 TFLOPS；BM=64 降到 32 次→~100 TFLOPS，接近
#     torch 114 TFLOPS）。
@tilelang.jit(out_idx=[3])
def draft_gemm_rtgrid(MAX_M16, N, K, dtype, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    """grid=(ceildiv(M,BLOCK_M), ceildiv(N,BLOCK_N))。M 是 runtime int32。
    X: [MAX_M16, K]（pad 行零填充），W: [N, K]，Out: [MAX_M16, N] = X @ W.T。"""
    accum = T.float32

    @T.prim_func
    def main(
        M: T.int32,
        X: T.Tensor([MAX_M16, K], dtype),
        W: T.Tensor([N, K], dtype),
        Out: T.Tensor([MAX_M16, N], dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (mb, nb):
            X_s = T.alloc_shared([BLOCK_M, BLOCK_K], dtype)
            W_s = T.alloc_shared([BLOCK_N, BLOCK_K], dtype)
            acc = T.alloc_fragment([BLOCK_M, BLOCK_N], accum)
            T.clear(acc)
            for kb in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                T.copy(X[mb * BLOCK_M:(mb + 1) * BLOCK_M, kb * BLOCK_K:(kb + 1) * BLOCK_K], X_s)
                T.copy(W[nb * BLOCK_N:(nb + 1) * BLOCK_N, kb * BLOCK_K:(kb + 1) * BLOCK_K], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                Out[mb * BLOCK_M + i, nb * BLOCK_N + j] = acc[i, j].astype(dtype)
    return main


# ============ cache + launcher ============
_kernel_cache: dict = {}
_pad_buf_cache: dict = {}

_TORCH_TO_TL = {
    torch.float16: T.float16,
    torch.bfloat16: T.bfloat16,
}


def _get_kernel(MAX_M16, N, K, dtype, BLOCK_M):
    key = (MAX_M16, N, K, dtype, BLOCK_M)
    if key not in _kernel_cache:
        _kernel_cache[key] = draft_gemm_rtgrid(
            MAX_M16, N, K, _TORCH_TO_TL[dtype], BLOCK_M=BLOCK_M)
    return _kernel_cache[key]


def _get_pad_buf(MAX_M16, K, dtype, device):
    key = (MAX_M16, K, dtype, device)
    if key not in _pad_buf_cache:
        _pad_buf_cache[key] = torch.zeros(MAX_M16, K, dtype=dtype, device=device)
    return _pad_buf_cache[key]


def draft_gemm(x, w, max_m):
    """x: [M, K] bf16（M 变长，M<=max_m），w: [N, K] bf16 → out: [M, N] = x @ w.T。

    max_m: 编译期 M 上界（query 路径 16，context 路径 4096）。
    BLOCK_M 按 max_m 选：query 路径（max_m<=16）用 16，context 路径用 64
    （BM=16 时大 M 下 W 被重复读 M/16 次，HBM 带宽浪费，见 kernel 注释）。
    """
    M, K = x.shape
    N = w.shape[0]
    # M 超过编译期上界（如 selector 的 num_reqs*N 变长）时回退 torch matmul，保证不崩。
    if M > max_m:
        return x @ w.T
    assert K % 128 == 0, f"draft_gemm: K={K} 必须 %128==0"
    assert N % 64 == 0, f"draft_gemm: N={N} 必须 %64==0"

    BLOCK_M = 16 if max_m <= 16 else 64
    MAX_M16 = (max_m + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    kern = _get_kernel(MAX_M16, N, K, x.dtype, BLOCK_M)
    xbuf = _get_pad_buf(MAX_M16, K, x.dtype, x.device)

    M16 = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    xbuf[:M].copy_(x)
    if M16 > M:
        xbuf[M:M16].zero_()

    out = kern(M, xbuf, w)
    return out[:M]
