"""Qwen3 decode TileLang 融合算子（tilert 思想：persistent + 数据不落地 + L2 复用）。

单层 decode 的算子链用 persistent kernel 融合，中间 tensor 留 shared mem/register 不落 HBM，
权重流式读与计算 pipeline overlap。M=16 零填充（mma.h 要求 M%16==0），bs=1 时 row0 真实、
rows1-15 恒零，只取 row0 输出。参考 pre_mla.py 的 persistent + phase 结构。

首版：SwiGLU MLP 全融合（gate_up GEMV + silu(gate)*up + down GEMV），不含 rmsnorm_residual。
  输入 normed h [16, hidden]（row0 真实），gu_w [hidden, 2*inter]，d_w [inter, hidden]
  输出 mlp_out [16, hidden]（row0 有效）
  中间 gate_up/act 分块留 shared mem，不落 HBM。
"""
import torch
import tilelang
import tilelang.language as T

_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def fused_swiglu_kernel(hidden, inter, dtype):
    """SwiGLU: out = (silu(gate) * up) @ d_w，gate_up = h @ gu_w（gu_w=[hidden,2*inter], up 前半 gate 后半）。
    M=16 pad，分块：gate_up 按 [16, BLK] 算，silu*up 后立即 down GEMV 累加，act 不落 HBM。
    down: out[16, hidden] = act[16, inter] @ d_w[inter, hidden]，按 inter 分块累加。"""
    accum = T.float32
    BLK = 128  # gate_up 输出分块大小
    N_GU = T.ceildiv(2 * inter, BLK)
    # down: hidden 维按 128 分块输出，inter 维全累加
    H_BLK = 128
    N_H = T.ceildiv(hidden, H_BLK)
    # inter 维 down 累加分块（与 gate_up 的 up/gate 半区对齐）
    I_BLK = 64
    N_I = T.ceildiv(inter, I_BLK)

    @T.prim_func
    def main(
        H16: T.Tensor([16, hidden], dtype),     # normed 输入 row0 真实
        GuW: T.Tensor([hidden, 2 * inter], dtype),  # [hidden, 2*inter] up 前半 gate 后半
        DW: T.Tensor([inter, hidden], dtype),   # down [inter, hidden]
        Out: T.Tensor([16, hidden], dtype),     # 输出 row0 有效
    ):
        # 每个 block 算输出 Out[0:16, hb*H_BLK:(hb+1)*H_BLK]
        with T.Kernel(N_H, threads=256) as (hb,):
            acc = T.alloc_fragment([16, H_BLK], accum)
            T.clear(acc)
            H_s = T.alloc_shared([16, 128], dtype)       # h 分块（复用）
            W_s = T.alloc_shared([128, 64], dtype)       # gu_w 分块 [hidden_blk, out_blk=64]
            up_s = T.alloc_shared([16, 64], dtype)        # act 结果（喂 down）
            acc_u = T.alloc_fragment([16, 64], accum)     # up 累加
            acc_g = T.alloc_fragment([16, 64], accum)     # gate 累加
            DW_s = T.alloc_shared([64, 128], dtype)       # d_w 分块 [I_BLK=64, H_BLK=128]

            for ib in range(N_I):
                # up = h @ gu_w[:, ib*64:(ib+1)*64]  (普通 GEMM: [16,128]@[128,64]→[16,64])
                T.clear(acc_u)
                for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                    T.copy(H16[0:16, kh * 128:(kh + 1) * 128], H_s)
                    T.copy(GuW[kh * 128:(kh + 1) * 128, ib * 64:(ib + 1) * 64], W_s)
                    T.gemm(H_s, W_s, acc_u, policy=T.GemmWarpPolicy.FullCol)
                # gate = h @ gu_w[:, inter+ib*64:inter+(ib+1)*64]
                T.clear(acc_g)
                for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                    T.copy(H16[0:16, kh * 128:(kh + 1) * 128], H_s)
                    T.copy(GuW[kh * 128:(kh + 1) * 128, inter + ib * 64:inter + (ib + 1) * 64], W_s)
                    T.gemm(H_s, W_s, acc_g, policy=T.GemmWarpPolicy.FullCol)
                # act = silu(gate) * up，写进 up_s 喂 down（acc_u=up, acc_g=gate 都在 fragment）
                for j in T.Parallel(16 * 64):
                    g = T.cast(acc_g[j // 64, j % 64], accum)
                    u = T.cast(acc_u[j // 64, j % 64], accum)
                    silu_g = g / (1.0 + T.exp(-g))
                    up_s[j // 64, j % 64] = T.cast(silu_g * u, dtype)
                # down: acc += act @ d_w[ib*64:(ib+1)*64, hb*H_BLK:(hb+1)*H_BLK]
                # [16,64] @ [64,128] → [16,128]
                T.copy(DW[ib * 64:(ib + 1) * 64, hb * H_BLK:(hb + 1) * H_BLK], DW_s)
                T.gemm(up_s, DW_s, acc, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, Out[0:16, hb * H_BLK:(hb + 1) * H_BLK])
    return main


_cache = {}


def get_fused_swiglu(hidden, inter, dtype):
    key = (hidden, inter, dtype)
    if key not in _cache:
        _cache[key] = fused_swiglu_kernel(hidden, inter, _TORCH_TO_TL[dtype])
    return _cache[key]


def fused_swiglu(h16, gu_w, d_w):
    """h16: [16, hidden] row0 真实；返回 [hidden]（row0）。"""
    hidden = h16.shape[1]
    inter = d_w.shape[0]
    kernel = get_fused_swiglu(hidden, inter, h16.dtype)
    out = kernel(h16, gu_w, d_w)
    return out[0]  # row0
