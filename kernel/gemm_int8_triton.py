"""投机解码 verify int8 GEMM 的开关 + 分派（TileLang 后端）。

历史注：本文件曾含 Triton int8 GEMM 后端（_triton_int8_gemm），因 Marlin 成为
verify GEMM 默认后端（adapter._MICRO_VERIFY_GEMM 默认 marlin）且实测 TileLang
快于 Triton（M=8 gate N=17408: 188.6 vs 196.1us）已删除，verify int8 GEMM
只剩 TileLang 一条路。
"""
import logging

logger = logging.getLogger(__name__)

# verify（M≤8）int8 GEMM 开关：由 SpecEngine 在 verify forward 前后设置
# （core/spec_decode.py）。开时 gemv_int8.w8_linear / adapter._lin_prefill 把
# group-128 int8 线性路由到 verify_int8_gemm（TileLang，权重 HBM 只读一次），
# 比 int8 GEMV 的 M× 权重读快 ~12x（mlp_gu 3.59ms→0.29ms）。
# 正常 decode（M=1）走 int8 GEMV、prefill（M 大）走反量化 matmul，均不受影响。
_verify_gemm_enabled = False


def set_verify_gemm(v: bool):
    global _verify_gemm_enabled
    _verify_gemm_enabled = v


def verify_gemm_enabled() -> bool:
    return _verify_gemm_enabled


def verify_int8_gemm(x, w_int8, scale, out=None):
    """verify int8 GEMM（group-128，TileLang）。x [M,K] bf16, w_int8 [N,K] int8,
    scale [N,K/128] fp32 → out [M,N] bf16。权重 HBM 只读一次（每 block 一个
    [BLOCK_N, BLOCK_K] tile），shared 内 dequant→bf16 后 T.gemm 累加。"""
    from kernel.gemm_int8 import int8_gemm
    return int8_gemm(x, w_int8, scale, out)
