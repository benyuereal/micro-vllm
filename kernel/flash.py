#!/usr/bin/env python3
"""
Paged decode attention — Triton kernels + Python wrapper.
Supports MHA / GQA, optional sliding-window, Split-KV for long sequences.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ── Helpers ────────────────────────────────────────────────────────────────────

@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


# ── Kernel: standard decode  (grid = [B, H]) ──────────────────────────────────

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
    kv_h = h // (num_heads // num_kv_heads) if USE_GQA else h

    ctx_len   = tl.load(context_lens_ptr + b)
    query_pos = ctx_len - 1
    offs_d    = tl.arange(0, head_size_padded)
    d_mask    = offs_d < head_size

    q = tl.load(query_ptr + b * stride_qb + h * stride_qh + offs_d,
                mask=d_mask, other=0.0)

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.full([], 0.0,           dtype=tl.float32)
    acc = tl.zeros([head_size_padded], dtype=tl.float32)

    for j in range(0, _cdiv(ctx_len, BLOCK_N)):
        offs_n  = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < ctx_len
        kv_d    = kv_mask[:, None] & d_mask[None, :]
        slot    = (offs_n % block_size)[:, None]
        phys    = tl.load(block_tables_ptr + b * stride_bt0 + offs_n // block_size,
                          mask=kv_mask, other=0).to(tl.int64)

        K = tl.load(k_cache_ptr + phys[:, None] * stride_kn + slot * stride_kb
                    + kv_h * stride_kh + offs_d[None, :] * stride_kd,
                    mask=kv_d, other=0.0)
        V = tl.load(v_cache_ptr + phys[:, None] * stride_vn + slot * stride_vb
                    + kv_h * stride_vh + offs_d[None, :] * stride_vd,
                    mask=kv_d, other=0.0)

        S = scale * tl.sum(q[None, :] * K, axis=1)
        causal = offs_n <= query_pos
        if SLIDING_WINDOW > 0:
            causal = causal & ((query_pos - offs_n) < SLIDING_WINDOW)
        S = tl.where(causal & kv_mask, S, float("-inf"))

        m_j   = tl.maximum(m_i, tl.max(S, axis=0))
        P     = tl.exp(S - m_j)
        alpha = tl.exp(m_i - m_j)
        l_i   = alpha * l_i + tl.sum(P, axis=0)
        acc   = alpha * acc + tl.sum(P.to(V.dtype)[:, None] * V, axis=0).to(tl.float32)
        m_i   = m_j

    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, acc / l_i, mask=d_mask)


# ── Kernel: Split-KV phase-1  (grid = [B, H, num_splits]) ─────────────────────

@triton.jit
def _splitkv_kernel(
        out_ptr, m_ptr, l_ptr,
        query_ptr, k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kn, stride_kb, stride_kh, stride_kd,
        stride_vn, stride_vb, stride_vh, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_mb, stride_mh, stride_ms,
        stride_lb, stride_lh, stride_ls,
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
    split_sz  = _cdiv(ctx_len, num_splits)
    kv_start  = s * split_sz
    kv_end    = tl.minimum(kv_start + split_sz, ctx_len)

    offs_d = tl.arange(0, head_size_padded)
    d_mask = offs_d < head_size

    q = tl.load(query_ptr + b * stride_qb + h * stride_qh + offs_d,
                mask=d_mask, other=0.0)

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.full([], 0.0,           dtype=tl.float32)
    acc = tl.zeros([head_size_padded], dtype=tl.float32)

    for j in range(0, _cdiv(kv_end - kv_start, BLOCK_N)):
        offs_n  = kv_start + j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < kv_end
        kv_d    = kv_mask[:, None] & d_mask[None, :]
        slot    = (offs_n % block_size)[:, None]
        phys    = tl.load(block_tables_ptr + b * stride_bt0 + offs_n // block_size,
                          mask=kv_mask, other=0).to(tl.int64)

        K = tl.load(k_cache_ptr + phys[:, None] * stride_kn + slot * stride_kb
                    + kv_h * stride_kh + offs_d[None, :] * stride_kd,
                    mask=kv_d, other=0.0)
        V = tl.load(v_cache_ptr + phys[:, None] * stride_vn + slot * stride_vb
                    + kv_h * stride_vh + offs_d[None, :] * stride_vd,
                    mask=kv_d, other=0.0)

        S = scale * tl.sum(q[None, :] * K, axis=1)
        causal = offs_n <= query_pos
        if SLIDING_WINDOW > 0:
            causal = causal & ((query_pos - offs_n) < SLIDING_WINDOW)
        S = tl.where(causal & kv_mask, S, float("-inf"))

        m_j   = tl.maximum(m_i, tl.max(S, axis=0))
        P     = tl.exp(S - m_j)
        alpha = tl.exp(m_i - m_j)
        l_i   = alpha * l_i + tl.sum(P, axis=0)
        acc   = alpha * acc + tl.sum(P.to(V.dtype)[:, None] * V, axis=0).to(tl.float32)
        m_i   = m_j

    tl.store(out_ptr + b * stride_ob + h * stride_oh + s * stride_os + offs_d, acc / l_i, mask=d_mask)
    tl.store(m_ptr  + b * stride_mb  + h * stride_mh  + s * stride_ms, m_i)
    tl.store(l_ptr  + b * stride_lb  + h * stride_lh  + s * stride_ls, l_i)


# ── Kernel: Split-KV phase-2 reduce  (grid = [B, H]) ──────────────────────────

@triton.jit
def _splitkv_reduce_kernel(
        out_ptr, part_out_ptr, part_m_ptr, part_l_ptr,
        stride_ob, stride_oh, stride_od,
        stride_pb, stride_ph, stride_ps, stride_pd,
        stride_mb, stride_mh, stride_ms,
        stride_lb, stride_lh, stride_ls,
        head_size:        tl.constexpr,
        head_size_padded: tl.constexpr,
        num_splits:       tl.constexpr,
):
    b = tl.program_id(0);  h = tl.program_id(1)
    offs_d = tl.arange(0, head_size_padded)
    d_mask = offs_d < head_size

    m_vals = tl.load(part_m_ptr + b * stride_mb + h * stride_mh + tl.arange(0, num_splits))
    l_vals = tl.load(part_l_ptr + b * stride_lb + h * stride_lh + tl.arange(0, num_splits))

    m_max = tl.max(m_vals)
    l_sum = tl.sum(l_vals * tl.exp(m_vals - m_max))

    acc = tl.zeros([head_size_padded], dtype=tl.float32)
    for s in range(num_splits):
        m_s = tl.load(part_m_ptr + b * stride_mb + h * stride_mh + s * stride_ms)
        o_s = tl.load(part_out_ptr + b * stride_pb + h * stride_ph + s * stride_ps + offs_d,
                      mask=d_mask, other=0.0)
        acc += o_s * tl.exp(m_s - m_max)

    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, acc / l_sum, mask=d_mask)


# ── Python wrapper ─────────────────────────────────────────────────────────────

class TritonDecodeAttention:
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
            block_size:     int   = 16,
            scale:          Optional[float] = None,
            sliding_window: int   = -1,
    ):
        self.num_heads        = num_heads
        self.num_kv_heads     = num_kv_heads
        self.head_size        = head_size
        self.block_size       = block_size
        self.scale            = scale or math.sqrt(head_size) ** -1
        self.sliding_window   = max(sliding_window, 0)  # 0 = disabled in kernel
        self.use_gqa          = num_heads != num_kv_heads
        self.head_size_padded = triton.next_power_of_2(head_size)
        self.BLOCK_N          = 64 if head_size <= 64 else 128

    def num_splits(self, seq_len: int) -> int:
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
        ns  = self.num_splits(int(context_lens.max()))

        # constexpr / launch args shared by both kernel variants
        # scale is passed positionally (before strides) so it is excluded here
        kw = dict(
            num_heads=H, num_kv_heads=self.num_kv_heads,
            head_size=D, head_size_padded=self.head_size_padded,
            block_size=self.block_size,
            BLOCK_N=self.BLOCK_N,
            USE_GQA=self.use_gqa,
            SLIDING_WINDOW=self.sliding_window,
            num_warps=4 if D <= 64 else 8,
            num_stages=2,
        )

        if ns == 1:
            _decode_kernel[(B, H)](
                out, query, key_cache, value_cache, block_tables, context_lens,
                self.scale,
                *query.stride(), *key_cache.stride(), *value_cache.stride(),
                *out.stride(), block_tables.stride(0),
                **kw,
            )
        else:
            part_o = torch.empty((B, H, ns, D), dtype=query.dtype,    device=query.device)
            part_m = torch.empty((B, H, ns),    dtype=torch.float32,  device=query.device)
            part_l = torch.empty((B, H, ns),    dtype=torch.float32,  device=query.device)

            _splitkv_kernel[(B, H, ns)](
                part_o, part_m, part_l,
                query, key_cache, value_cache, block_tables, context_lens,
                self.scale,
                *query.stride(), *key_cache.stride(), *value_cache.stride(),
                *part_o.stride(), *part_m.stride(), *part_l.stride(),
                block_tables.stride(0),
                num_splits=ns, **kw,
            )
            _splitkv_reduce_kernel[(B, H)](
                out, part_o, part_m, part_l,
                *out.stride(), *part_o.stride(), *part_m.stride(), *part_l.stride(),
                head_size=D, head_size_padded=self.head_size_padded,
                num_splits=ns, num_warps=4,
            )

        return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Smoke test — TritonDecodeAttention")
    attn = TritonDecodeAttention(num_heads=32, num_kv_heads=32, head_size=128)
    B, H, D = 4, 32, 128
    q  = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    kc = torch.randn(100, 16, H, D, device="cuda", dtype=torch.float16)
    vc = torch.randn(100, 16, H, D, device="cuda", dtype=torch.float16)
    bt = torch.randint(0, 100, (B, 256), device="cuda", dtype=torch.int32)
    cl = torch.full((B,), 1024, device="cuda", dtype=torch.int32)
    out = attn.forward(q, kc, vc, bt, cl)
    print(f"Output shape: {out.shape}  OK")
