"""Triton int8 GEMM（group-128）+ 投机解码 verify 双后端分派。

背景：verify（M=1+N≈8）原走手写 CUDA tiled GEMV（gemv_int8_group_tiled_kernel），
compute-bound 3.59ms/层（mlp_gu），慢 12x。五路径 benchmark（M=8，同一份 int8 权重，
mlp_gu N=34816 K=5120 W=178MB）：
  - Triton int8 GEMM: 0.289ms（NVIDIA+ROCm 双平台可编译）
  - TileLang int8 GEMM: 0.292ms（NVIDIA 已验证）
  - CUDA tiled GEMV: 3.590ms（已删除）
  - 8×M=1 GEMV: 3.604ms
  - cuBLAS bf16: 0.498ms
本模块提供 Triton 后端（kernel 经 /tmp/bench_int8_paths.py 验证，maxdiff vs 反量化
matmul ~1% 相对误差，正常 bf16 舍入）+ verify 双后端分派（MICRO_VERIFY_GEMM）。

数值说明：int8 权重 shared 内 dequant→bf16 后 tl.dot（bf16 乘、fp32 累加），与
反量化 matmul 有 ~1% 相对误差（bf16 舍入），不再 bit-exact 原 int8 GEMV——用户已
接受（正确性标准：spec 输出合理 + acceptance >6/7）。
"""
import os
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _triton_int8_gemm(X, W, S, Out, M, N, K,
                      sxm, sxk, swn, swk, ssn, ssk, som, son,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k0 = kb * BLOCK_K
        xm = (om[:, None] < M) & (k0 + ok[None, :] < K)
        x = tl.load(X + om[:, None]*sxm + (k0+ok[None,:])*sxk, mask=xm, other=0.0)
        wm = (on[:, None] < N) & (k0 + ok[None, :] < K)
        w = tl.load(W + on[:, None]*swn + (k0+ok[None,:])*swk, mask=wm, other=0).to(tl.float32)
        s = tl.load(S + on*ssn + kb*ssk, mask=on < N, other=0.0)
        wb = (w * s[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, wb.T, acc)
    omask = (om[:, None] < M) & (on[None, :] < N)
    tl.store(Out + om[:, None]*som + on[None,:]*son, acc.to(tl.bfloat16), mask=omask)


def int8_gemm_triton(x, w_int8, scale, out=None):
    """W8A16 int8 GEMM（Triton，group-128）。x [M,K] bf16, w_int8 [N,K] int8,
    scale [N,K/128] fp32 → out [M,N] bf16。M 小（verify M=1+N≈8）时权重 HBM 只读一次。"""
    M, K = x.shape
    N = w_int8.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM = max(16, triton.next_power_of_2(M))
    # BN=64 在 M=8(verify)/61/128(prefill) 全比 BN=128 快（FLUSH=1 冷读实测，
    # gate N=17408 M=8: 196.1 vs 200.7us；M=128: 269 vs 327us）。BK=128 必须=group
    # 尺寸（scale 每 128 元素一个，BK=256 会跨 group 读错 scale，且 per-element
    # scale 会物化整块 fp32 scale tile → shared OOM）。
    BN = 64
    BK = 128
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _triton_int8_gemm[grid](x, w_int8, scale, out, M, N, K,
                            x.stride(0), x.stride(1), w_int8.stride(0), w_int8.stride(1),
                            scale.stride(0), scale.stride(1), out.stride(0), out.stride(1),
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=8, num_stages=3)
    return out


# ---------------------------------------------------------------------------
# verify（M≤8）双后端分派
# ---------------------------------------------------------------------------
# 投机解码 verify forward（M=1+N≈8）开关：由 SpecEngine 在 verify 前后
# 设置。开时 adapter._lin_prefill 把 group-128 int8 路由到 verify_int8_gemm
# （TileLang 默认 / Triton 备选），权重 HBM 只读一次（shared 内 dequant→bf16 +
# GEMM），比原 CUDA tiled GEMV 快 ~12x（mlp_gu 3.59ms→0.29ms）。
# 正常 decode（M=1）走 int8 GEMV、prefill（M 大）走反量化 matmul，均不受影响。
_verify_gemm_enabled = False
_backend = None


def set_verify_gemm(v: bool):
    global _verify_gemm_enabled
    _verify_gemm_enabled = v


def verify_gemm_enabled() -> bool:
    return _verify_gemm_enabled


def _select_backend():
    """选 verify int8 GEMM 后端（缓存）。

    MICRO_VERIFY_GEMM=tilelang|triton，默认 tilelang（约束：tile op 用 TileLang，
    Triton 仅 ROCm fallback / pointwise 小算子）；ROCm（torch.version.hip 非空）
    或 tilelang import 失败时 fallback triton（Triton 双平台可编译）。

    默认 tilelang 的依据（M=8 verify, FLUSH=1 HBM 冷读, 受控对比 median-of-5）：
    优化后 TileLang（_int8_gemm_kernel_fast：int8 cp.async 直进 shared + dequant 到
    bf16 shared + BK=256，gate_proj 203→189us 逼近 Marlin 174.8）在最大层反超
    Triton：
      gate N=17408:  tilelang 188.6 vs triton 196.1us
      q_proj N=12288: tilelang 155.2 vs triton 143.1us
      in_proj N=10240: tilelang 118.8 vs triton 117.8us
    两者均 ~1.08-1.23x 慢于 Marlin（174.1/126.0/106.5us），差距来自 Marlin 把
    dequant 增量融进 mma 循环（逐 fragment 交错，与 HBM 读完全 overlap），TileLang
    T.gemm 先 dequant 整 tile 再 mma 无法交错——结构性，非实现问题。
    注：verify 真实默认后端是 Marlin（adapter._MICRO_VERIFY_GEMM 默认 marlin），
    本分派仅在 MICRO_VERIFY_GEMM=tilelang|triton 显式指定时生效。"""
    global _backend
    if _backend is not None:
        return _backend
    want = os.environ.get("MICRO_VERIFY_GEMM", "tilelang").lower()
    if torch.version.hip is not None:
        want = "triton"
    if want == "tilelang":
        try:
            from kernel.gemm_int8 import int8_gemm
            _backend = ("tilelang", int8_gemm)
            logger.info("verify int8 GEMM 后端: TileLang")
            return _backend
        except Exception as e:
            logger.warning(f"TileLang int8 GEMM 不可用，fallback Triton: {e}")
    _backend = ("triton", int8_gemm_triton)
    logger.info("verify int8 GEMM 后端: Triton")
    return _backend


def verify_int8_gemm(x, w_int8, scale, out=None):
    """verify（M≤8）int8 GEMM（group-128）：TileLang 默认 / Triton 备选。
    x [M,K] bf16, w_int8 [N,K] int8, scale [N,K/128] fp32 → out [M,N] bf16。"""
    _, fn = _select_backend()
    return fn(x, w_int8, scale, out)
