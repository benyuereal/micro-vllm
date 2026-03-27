#!/usr/bin/env python3
"""
Paged decode attention — Triton kernels + Python wrapper.
Supports MHA / GQA, optional sliding-window, Split-KV for long sequences.

算法概览
--------
标准注意力需要将完整的 softmax 权重矩阵 P [B, H, 1, N] 写入显存再读回，
Flash Attention 用「在线 softmax」消除了这次 HBM 往返：

    初始化：cur_max = -∞,  cur_exp_sum = 0,  acc = 0
    对每个 KV tile（BLOCK_N 个 token）：
        S        = scale * Q @ K^T          # 注意力分数
        block_max = max(cur_max, max(S))
        rescale  = exp(cur_max - block_max) # 旧统计量的校正因子
        P        = exp(S - block_max)
        cur_exp_sum = rescale * cur_exp_sum + sum(P)   # 更新 softmax 分母
        acc         = rescale * acc + P @ V            # 更新加权输出
        cur_max     = block_max
    输出：acc / cur_exp_sum

对于超长序列（>8K token），Split-KV 将 KV 分成若干段并行计算，
再用相同的在线 softmax 原理把各段结果归约合并。
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ── Helpers ────────────────────────────────────────────────────────────────────

@triton.jit
def _cdiv(x, y):
    # 向上取整除法，等价于 math.ceil(x / y)
    return (x + y - 1) // y


# ── Shared inner loop: online softmax over a [kv_start, kv_end) tile range ─────
# 两个主 kernel（标准 decode 和 Split-KV phase-1）共用同一套在线 softmax 循环体，
# 唯一区别是迭代范围：标准 decode 传 [0, ctx_len)，Split-KV 传各自的分段范围。

@triton.jit
def _attend_tiles(
        q, k_cache_ptr, v_cache_ptr, block_tables_ptr,
        scale, query_pos, kv_start, kv_end,
        cur_max, cur_exp_sum, acc,
        kv_h, b, offs_d, d_mask,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_bt0,
        block_size:     tl.constexpr,  # KV cache 分页的页大小（token 数）
        BLOCK_N:        tl.constexpr,  # 每次处理的 KV tile 大小（token 数）
        SLIDING_WINDOW: tl.constexpr,  # 滑动窗口大小，0 表示不启用
):
    # 对 [kv_start, kv_end) 范围内的 KV token 逐 tile 做在线 softmax：
    #
    # 分页地址：逻辑位置 offs_n → 物理块号 block_idx = block_tables[b, offs_n // block_size]
    #           块内偏移 slot = offs_n % block_size
    #
    # 每轮 tile 更新（在线 softmax 核心公式）：
    #   S           = scale * Q @ K^T           # 注意力分数 [BLOCK_N]
    #   block_max   = max(cur_max, max(S))       # 更新全局最大值
    #   rescale     = exp(cur_max - block_max)   # 旧统计量的校正因子，block_max 变大时 rescale < 1
    #   P           = exp(S - block_max)         # 当前 tile 未归一化权重
    #   cur_exp_sum = rescale * cur_exp_sum + sum(P)       # 更新 softmax 分母
    #   acc         = rescale * acc + P @ V                # 更新加权输出累积
    #
    # 因果掩码：offs_n <= query_pos，超范围位置 S 置 -inf（exp 后贡献为零）
    # 滑动窗口：额外限制 query_pos - offs_n < SLIDING_WINDOW
    num_tiles = _cdiv(kv_end - kv_start, BLOCK_N)
    for j in range(0, num_tiles):
        offs_n    = kv_start + j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask   = offs_n < kv_end
        kv_d      = kv_mask[:, None] & d_mask[None, :]

        slot      = (offs_n % block_size)[:, None]
        block_idx = tl.load(block_tables_ptr + b * stride_bt0 + offs_n // block_size,
                            mask=kv_mask, other=0).to(tl.int64)

        K = tl.load(k_cache_ptr + block_idx[:, None] * stride_kn + slot * stride_kb
                    + kv_h * stride_kh + offs_d[None, :] * stride_kd,
                    mask=kv_d, other=0.0)
        V = tl.load(v_cache_ptr + block_idx[:, None] * stride_vn + slot * stride_vb
                    + kv_h * stride_vh + offs_d[None, :] * stride_vd,
                    mask=kv_d, other=0.0)

        S = scale * tl.sum(q[None, :] * K, axis=1)

        causal = offs_n <= query_pos
        if SLIDING_WINDOW > 0:
            causal = causal & ((query_pos - offs_n) < SLIDING_WINDOW)
        S = tl.where(causal & kv_mask, S, float("-inf"))

        block_max   = tl.maximum(cur_max, tl.max(S, axis=0))
        P           = tl.exp(S - block_max)
        rescale     = tl.exp(cur_max - block_max)
        cur_exp_sum = rescale * cur_exp_sum + tl.sum(P, axis=0)
        acc         = rescale * acc + tl.sum(P.to(V.dtype)[:, None] * V, axis=0).to(tl.float32)
        cur_max     = block_max

    return cur_max, cur_exp_sum, acc


# ── Kernel: standard decode  (grid = [B, H]) ──────────────────────────────────
# 短序列路径（seq_len ≤ 8192）：单个线程一次扫完全部 KV，直接写出最终结果。
# grid 维度：axis-0 = batch index，axis-1 = query head index。

@triton.jit
def _decode_kernel(
        out_ptr,
        query_ptr, k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_ob, stride_oh, stride_od,
        stride_bt0,
        num_heads:        tl.constexpr,
        num_kv_heads:     tl.constexpr,
        head_size:        tl.constexpr,
        head_size_padded: tl.constexpr,
        block_size:       tl.constexpr,
        BLOCK_N:          tl.constexpr,
        USE_GQA:          tl.constexpr,
        SLIDING_WINDOW:   tl.constexpr,
):
    b = tl.program_id(0);  h = tl.program_id(1)
    # GQA：num_heads / num_kv_heads 个 query head 共享同一组 KV head
    kv_h = h // (num_heads // num_kv_heads) if USE_GQA else h

    ctx_len   = tl.load(context_lens_ptr + b)
    query_pos = ctx_len - 1
    offs_d    = tl.arange(0, head_size_padded)
    d_mask    = offs_d < head_size

    q = tl.load(query_ptr + b * stride_qb + h * stride_qh + offs_d,
                mask=d_mask, other=0.0)

    cur_max, cur_exp_sum, acc = _attend_tiles(
        q, k_cache_ptr, v_cache_ptr, block_tables_ptr,
        scale, query_pos, 0, ctx_len,
        tl.full([], float("-inf"), dtype=tl.float32),
        tl.full([], 0.0,           dtype=tl.float32),
        tl.zeros([head_size_padded], dtype=tl.float32),
        kv_h, b, offs_d, d_mask,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_bt0, block_size, BLOCK_N, SLIDING_WINDOW,
    )
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, acc / cur_exp_sum, mask=d_mask)


# ── Kernel: Split-KV phase-1  (grid = [B, H, num_splits]) ─────────────────────
# 长序列路径的第一阶段：将 KV 序列均匀分成 num_splits 段，每个线程只处理自己的段。
# grid 增加了第三维 s = split index，各段完全并行，互不依赖。
# 每个线程输出三个中间量（供 phase-2 归约）：
#   split_acc   [B, H, S, D]  — 局部归一化输出：acc_s / cur_exp_sum_s
#   split_max   [B, H, S]     — 局部最大值 cur_max_s（归约时用于数值稳定缩放）
#   split_denom [B, H, S]     — 局部 softmax 分母 cur_exp_sum_s

@triton.jit
def _splitkv_kernel(
        split_acc_ptr, split_max_ptr, split_denom_ptr,
        query_ptr, k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_sacc_b, stride_sacc_h, stride_sacc_s, stride_sacc_d,
        stride_smax_b, stride_smax_h, stride_smax_s,
        stride_sdenom_b, stride_sdenom_h, stride_sdenom_s,
        stride_bt0,
        num_heads:        tl.constexpr,
        num_kv_heads:     tl.constexpr,
        head_size:        tl.constexpr,
        head_size_padded: tl.constexpr,
        block_size:       tl.constexpr,
        num_splits:       tl.constexpr,
        BLOCK_N:          tl.constexpr,
        USE_GQA:          tl.constexpr,
        SLIDING_WINDOW:   tl.constexpr,
):
    b = tl.program_id(0);  h = tl.program_id(1);  s = tl.program_id(2)
    kv_h = h // (num_heads // num_kv_heads) if USE_GQA else h

    ctx_len   = tl.load(context_lens_ptr + b)
    query_pos = ctx_len - 1
    # 均匀切分：每段向上取整，最后一段截断到 ctx_len
    split_sz  = _cdiv(ctx_len, num_splits)
    kv_start  = s * split_sz
    kv_end    = tl.minimum(kv_start + split_sz, ctx_len)

    offs_d = tl.arange(0, head_size_padded)
    d_mask = offs_d < head_size

    q = tl.load(query_ptr + b * stride_qb + h * stride_qh + offs_d,
                mask=d_mask, other=0.0)

    cur_max, cur_exp_sum, acc = _attend_tiles(
        q, k_cache_ptr, v_cache_ptr, block_tables_ptr,
        scale, query_pos, kv_start, kv_end,
        tl.full([], float("-inf"), dtype=tl.float32),
        tl.full([], 0.0,           dtype=tl.float32),
        tl.zeros([head_size_padded], dtype=tl.float32),
        kv_h, b, offs_d, d_mask,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_bt0, block_size, BLOCK_N, SLIDING_WINDOW,
    )
    tl.store(split_acc_ptr   + b * stride_sacc_b  + h * stride_sacc_h  + s * stride_sacc_s  + offs_d, acc / cur_exp_sum, mask=d_mask)
    tl.store(split_max_ptr   + b * stride_smax_b  + h * stride_smax_h  + s * stride_smax_s,  cur_max)
    tl.store(split_denom_ptr + b * stride_sdenom_b + h * stride_sdenom_h + s * stride_sdenom_s, cur_exp_sum)


# ── Kernel: Split-KV phase-2 reduce  (grid = [B, H]) ──────────────────────────
# 第二阶段：将 phase-1 各段的局部结果合并为全局输出。
# 合并公式（与在线 softmax 原理相同）：
#   global_max     = max(split_max_0, ..., split_max_S)
#   global_exp_sum = Σᵢ exp(split_max_i - global_max) × split_denom_i
#   O = Σᵢ [exp(split_max_i - global_max) × split_acc_i] / global_exp_sum
#
# 注意 split_acc_i 存的是已归一化的局部输出（acc_i / denom_i），
# 合并时先乘回 exp * denom 还原未归一化贡献，最后统一除以 global_exp_sum。

@triton.jit
def _splitkv_reduce_kernel(
        out_ptr, split_acc_ptr, split_max_ptr, split_denom_ptr,
        stride_out_b, stride_out_h, stride_out_d,
        stride_sacc_b, stride_sacc_h, stride_sacc_s, stride_sacc_d,
        stride_smax_b, stride_smax_h, stride_smax_s,
        stride_sdenom_b, stride_sdenom_h, stride_sdenom_s,
        head_size:        tl.constexpr,
        head_size_padded: tl.constexpr,
        num_splits:       tl.constexpr,
):
    b = tl.program_id(0);  h = tl.program_id(1)
    offs_d = tl.arange(0, head_size_padded)
    d_mask = offs_d < head_size

    split_maxes    = tl.load(split_max_ptr   + b * stride_smax_b   + h * stride_smax_h   + tl.arange(0, num_splits))
    split_exp_sums = tl.load(split_denom_ptr + b * stride_sdenom_b + h * stride_sdenom_h + tl.arange(0, num_splits))

    global_max     = tl.max(split_maxes)
    global_exp_sum = tl.sum(split_exp_sums * tl.exp(split_maxes - global_max))

    acc = tl.zeros([head_size_padded], dtype=tl.float32)
    for s in range(num_splits):
        split_max = tl.load(split_max_ptr  + b * stride_smax_b  + h * stride_smax_h  + s * stride_smax_s)
        split_out = tl.load(split_acc_ptr  + b * stride_sacc_b  + h * stride_sacc_h  + s * stride_sacc_s  + offs_d,
                            mask=d_mask, other=0.0)
        acc += split_out * tl.exp(split_max - global_max)

    tl.store(out_ptr + b * stride_out_b + h * stride_out_h + offs_d, acc / global_exp_sum, mask=d_mask)


# ── Python wrapper ─────────────────────────────────────────────────────────────

class FlashAttention:
    """
    Single-token decode attention over a paged KV cache.
    Supports MHA / GQA, optional sliding window.
    Automatically switches to Split-KV for long sequences.
    """

    def __init__(
            self,
            num_heads:      int   = 32,
            num_kv_heads:   int   = 32,
            head_size:      int   = 128,
            block_size:     int   = 16,   # KV cache 分页大小（每页存多少个 token）
            scale:          Optional[float] = None,
            sliding_window: int   = -1,
    ):
        self.num_heads        = num_heads
        self.num_kv_heads     = num_kv_heads
        self.head_size        = head_size
        self.block_size       = block_size
        self.scale            = scale or math.sqrt(head_size) ** -1  # 默认 1/√D
        self.sliding_window   = max(sliding_window, 0)  # 0 = disabled in kernel
        self.use_gqa          = num_heads != num_kv_heads
        # Triton 要求 tile 维度是 2 的幂，将 head_size 向上对齐
        self.head_size_padded = triton.next_power_of_2(head_size)
        # 每次处理的 KV token 数（在线 softmax 的 tile 大小）
        # 注意：这与 block_size（KV cache 的分页大小）是两个不同的概念
        self.kv_tile_size     = 64 if head_size <= 64 else 128

    def num_splits(self, seq_len: int) -> int:
        # 根据序列长度决定分段数：短序列不分段，避免 kernel launch 开销；
        # 超长序列增大分段数以充分利用 GPU 并行度
        if seq_len <= 8192:  return 1
        if seq_len <= 32768: return 2
        return 4

    # Alias for backward compat
    def get_num_splits(self, seq_len: int) -> int:
        return self.num_splits(seq_len)

    def forward(
            self,
            query:        torch.Tensor,   # [B, H, D]
            key_cache:    torch.Tensor,   # [num_blocks, block_size, H_kv, D]
            value_cache:  torch.Tensor,
            block_tables: torch.Tensor,   # [B, max_blocks]
            context_lens: torch.Tensor,   # [B]  int32
    ) -> torch.Tensor:
        B, H, D = query.shape
        assert H == self.num_heads and D == self.head_size

        out = torch.empty_like(query)
        # 按 batch 中最长序列决定是否启用 Split-KV
        num_splits = self.num_splits(int(context_lens.max()))

        # 两个 kernel 共用的 constexpr 启动参数
        # scale 通过位置参数传递（在 strides 之前），因此不放入此 dict
        kernel_kwargs = dict(
            num_heads=H, num_kv_heads=self.num_kv_heads,
            head_size=D, head_size_padded=self.head_size_padded,
            block_size=self.block_size,
            BLOCK_N=self.kv_tile_size,
            USE_GQA=self.use_gqa,
            SLIDING_WINDOW=self.sliding_window,
            num_warps=4 if D <= 64 else 8,
            num_stages=2,
        )

        if num_splits == 1:
            _decode_kernel[(B, H)](
                out, query, key_cache, value_cache, block_tables, context_lens,
                self.scale,
                *query.stride(), *key_cache.stride(), *value_cache.stride(),
                *out.stride(), block_tables.stride(0),
                **kernel_kwargs,
            )
        else:
            split_acc   = torch.empty((B, H, num_splits, D), dtype=query.dtype,   device=query.device)
            split_max   = torch.empty((B, H, num_splits),    dtype=torch.float32, device=query.device)
            split_denom = torch.empty((B, H, num_splits),    dtype=torch.float32, device=query.device)

            _splitkv_kernel[(B, H, num_splits)](
                split_acc, split_max, split_denom,
                query, key_cache, value_cache, block_tables, context_lens,
                self.scale,
                *query.stride(), *key_cache.stride(), *value_cache.stride(),
                *split_acc.stride(), *split_max.stride(), *split_denom.stride(),
                block_tables.stride(0),
                num_splits=num_splits, **kernel_kwargs,
            )
            _splitkv_reduce_kernel[(B, H)](
                out, split_acc, split_max, split_denom,
                *out.stride(), *split_acc.stride(), *split_max.stride(), *split_denom.stride(),
                head_size=D, head_size_padded=self.head_size_padded,
                num_splits=num_splits, num_warps=4,
            )

        return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke test — TritonDecodeAttention")
    attn = FlashAttention(num_heads=32, num_kv_heads=32, head_size=128)
    B, H, D = 4, 32, 128
    query       = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    key_cache   = torch.randn(100, 16, H, D, device="cuda", dtype=torch.float16)
    value_cache = torch.randn(100, 16, H, D, device="cuda", dtype=torch.float16)
    block_tables = torch.randint(0, 100, (B, 256), device="cuda", dtype=torch.int32)
    context_lens = torch.full((B,), 1024, device="cuda", dtype=torch.int32)
    out = attn.forward(query, key_cache, value_cache, block_tables, context_lens)
    print(f"Output shape: {out.shape}  OK")
