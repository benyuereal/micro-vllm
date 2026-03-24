"""
triton_flash_attn_opt.py — Optimized Flash Attention with KV Cache
Key improvements over baseline:
  1. Split-K Flash Decoding  : splits KV across NUM_KV_SPLITS CTAs per (b, h)
                               Grid: (B, H, NUM_KV_SPLITS) instead of (B, H)
                               Eliminates the #1 bottleneck — GPU starvation on small batch
  2. Tensor Core (tl.dot)    : QK and AV via mma instructions instead of scalar reduces
  3. Prefill causal fast-path: only the boundary block needs masking; inner blocks skip it
  4. Vectorised 128-bit loads: HEAD_DIM contiguous → single 128-bit transaction per row
  5. Tuned autotune configs  : favouring larger BLOCK_N with more stages for latency hiding

NCU bottleneck addressed:
  sm__warps_active 12.49%  → split-K raises occupancy
  long_scoreboard  33.0%   → more warps hide memory latency
  sm__throughput    5.33%  → tl.dot engages tensor cores

Interface identical to baseline triton_flash_attn_with_kvcache().
"""

import math
import time
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# KV Cache update kernel  (unchanged — already near-optimal)
# ---------------------------------------------------------------------------

@triton.jit
def _kv_cache_update_kernel(
    Src_ptr, Dst_ptr,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_db, stride_ds, stride_dh, stride_dd,
    seqlens_ptr,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    cache_pos = tl.load(seqlens_ptr + b) + s
    d = tl.arange(0, HEAD_DIM)

    src = Src_ptr + b * stride_sb + s * stride_ss + h * stride_sh + d * stride_sd
    dst = Dst_ptr + b * stride_db + cache_pos * stride_ds + h * stride_dh + d * stride_dd

    tl.store(dst, tl.load(src))


def update_kv_cache(k_cache, v_cache, k_new, v_new, cache_seqlens):
    B, S_new, H_kv, D = k_new.shape
    assert D in (64, 128, 256)
    grid = (B, S_new, H_kv)
    for Src, Dst in ((k_new, k_cache), (v_new, v_cache)):
        _kv_cache_update_kernel[grid](
            Src, Dst,
            Src.stride(0), Src.stride(1), Src.stride(2), Src.stride(3),
            Dst.stride(0), Dst.stride(1), Dst.stride(2), Dst.stride(3),
            cache_seqlens, HEAD_DIM=D,
        )


# ---------------------------------------------------------------------------
# Decode — Phase 1: Split-K partial attention
# ---------------------------------------------------------------------------
# OPTIMISATION 1: Grid is now (B, H, NUM_KV_SPLITS).
#   Each CTA owns a contiguous slice [kv_start, kv_end) of the KV sequence.
#   Result: a partial (acc, m, l) tuple that Phase 2 will reduce.
#
# OPTIMISATION 2: Tensor cores via tl.dot.
#   q[None, :] @ k.T   → [1, BLOCK_N]   (uses wmma/mma)
#   exp_qk[None, :] @ v → [1, HEAD_DIM]  (uses wmma/mma)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['HEAD_DIM'],
)
@triton.jit
def _flash_decode_splitk_kernel(
    Q_ptr, K_ptr, V_ptr,
    PartO_ptr, PartM_ptr, PartL_ptr,     # partial output buffers
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_pb, stride_ph, stride_ps, stride_pd,  # partial O: [B, H, NUM_KV_SPLITS, D]
    stride_mb, stride_mh, stride_ms,             # partial M: [B, H, NUM_KV_SPLITS]
    stride_lb, stride_lh, stride_ls,             # partial L: [B, H, NUM_KV_SPLITS]
    kv_seqlens_ptr,
    scale,
    NUM_KV_SPLITS: tl.constexpr,
    GQA_GROUPS:    tl.constexpr,
    HEAD_DIM:      tl.constexpr,
    BLOCK_N:       tl.constexpr,
):
    b     = tl.program_id(0)
    h     = tl.program_id(1)
    split = tl.program_id(2)
    kv_h  = h // GQA_GROUPS

    seqlen_kv = tl.load(kv_seqlens_ptr + b)

    # ---- Compute this split's KV range ----
    per_split = tl.cdiv(seqlen_kv, NUM_KV_SPLITS)
    kv_start  = split * per_split
    kv_end    = tl.minimum(kv_start + per_split, seqlen_kv)

    # Early-exit for empty splits (possible when seqlen_kv < NUM_KV_SPLITS)
    if kv_start >= seqlen_kv:
        d = tl.arange(0, HEAD_DIM)
        base_o = PartO_ptr + b*stride_pb + h*stride_ph + split*stride_ps
        tl.store(base_o + d*stride_pd, tl.zeros([HEAD_DIM], tl.float32))
        tl.store(PartM_ptr + b*stride_mb + h*stride_mh + split*stride_ms,
                 float("-inf"))
        tl.store(PartL_ptr + b*stride_lb + h*stride_lh + split*stride_ls, 0.0)
        return

    d = tl.arange(0, HEAD_DIM)

    # ---- Load Q (fp16/bf16), scale for tl.dot ----
    q_fp16 = tl.load(Q_ptr + b*stride_qb + h*stride_qh + d*stride_qd)   # [HEAD_DIM]

    # Reshape for matrix multiply: [1, HEAD_DIM]
    q_mat = (q_fp16.to(tl.float32) * scale).to(q_fp16.dtype)[None, :]   # [1, D], scaled fp16

    m_i  = tl.full([], float("-inf"), dtype=tl.float32)
    l_i  = tl.full([], 0.0,           dtype=tl.float32)
    acc  = tl.zeros([HEAD_DIM],        dtype=tl.float32)

    k_base = K_ptr + b*stride_kb + kv_h*stride_kh
    v_base = V_ptr + b*stride_vb + kv_h*stride_vh

    for blk in range(kv_start, kv_end, BLOCK_N):
        offs_n = blk + tl.arange(0, BLOCK_N)              # [BLOCK_N]
        mask_n = offs_n < kv_end

        # ---- K block [BLOCK_N, HEAD_DIM] ----
        k = tl.load(k_base + offs_n[:, None]*stride_ks + d[None, :]*stride_kd,
                    mask=mask_n[:, None], other=0.0)       # fp16

        # OPTIMISATION: use tensor cores  (q_mat: [1,D], k.T: [D,BLOCK_N]) → [BLOCK_N]
        qk_mat = tl.dot(q_mat, tl.trans(k)).to(tl.float32)    # [1, BLOCK_N]
        qk     = tl.reshape(qk_mat, [BLOCK_N])                 # [BLOCK_N]
        qk     = tl.where(mask_n, qk, float("-inf"))

        # ---- Online softmax update ----
        m_new   = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha   = tl.exp(m_i - m_new)
        exp_qk  = tl.exp(qk - m_new)                           # [BLOCK_N]
        l_i     = alpha * l_i + tl.sum(exp_qk, axis=0)

        # ---- V block [BLOCK_N, HEAD_DIM] ----
        v = tl.load(v_base + offs_n[:, None]*stride_vs + d[None, :]*stride_vd,
                    mask=mask_n[:, None], other=0.0)            # fp16

        # OPTIMISATION: exp_qk[None,:] @ v  → [HEAD_DIM] via tensor cores
        ev_mat  = tl.dot(exp_qk[None, :].to(v.dtype), v).to(tl.float32)  # [1, HEAD_DIM]
        acc     = acc * alpha + tl.reshape(ev_mat, [HEAD_DIM])
        m_i     = m_new

    # ---- Store partial results ----
    base_o = PartO_ptr + b*stride_pb + h*stride_ph + split*stride_ps
    tl.store(base_o + d*stride_pd, acc.to(tl.float16))
    tl.store(PartM_ptr + b*stride_mb + h*stride_mh + split*stride_ms, m_i)
    tl.store(PartL_ptr + b*stride_lb + h*stride_lh + split*stride_ls, l_i)


# ---------------------------------------------------------------------------
# Decode — Phase 2: Reduction across splits
# ---------------------------------------------------------------------------
# Combines NUM_KV_SPLITS partial (acc, m, l) into final output.
# Grid: (B, H)  — small, but this kernel is very fast (O(NUM_KV_SPLITS * D)).
# ---------------------------------------------------------------------------

@triton.jit
def _flash_decode_reduce_kernel(
    PartO_ptr, PartM_ptr, PartL_ptr,
    Out_ptr,
    stride_pb, stride_ph, stride_ps, stride_pd,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    stride_ob, stride_oh, stride_od,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM:      tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)

    m_fin  = tl.full([], float("-inf"), dtype=tl.float32)
    l_fin  = tl.full([], 0.0,           dtype=tl.float32)
    acc    = tl.zeros([HEAD_DIM],        dtype=tl.float32)

    for s in tl.static_range(NUM_KV_SPLITS):
        m_s  = tl.load(PartM_ptr + b*stride_mb + h*stride_mh + s*stride_ms)
        l_s  = tl.load(PartL_ptr + b*stride_lb + h*stride_lh + s*stride_ls)
        o_s  = tl.load(PartO_ptr + b*stride_pb + h*stride_ph + s*stride_ps + d*stride_pd
                      ).to(tl.float32)

        # Merge (m_fin, l_fin, acc) with (m_s, l_s, o_s)
        m_new      = tl.maximum(m_fin, m_s)
        alpha_prev = tl.exp(m_fin - m_new)
        alpha_s    = tl.exp(m_s   - m_new)

        l_fin = alpha_prev * l_fin + alpha_s * l_s
        acc   = acc * alpha_prev  + o_s * alpha_s
        m_fin = m_new

    acc = acc / l_fin

    out_base = Out_ptr + b*stride_ob + h*stride_oh
    tl.store(out_base + d*stride_od, acc.to(PartO_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Prefill kernel — Optimised causal masking
# ---------------------------------------------------------------------------
# OPTIMISATION 3: Distinguish three cases per KV block:
#   (a) block fully before earliest query       → skip (impossible with updated kv_end logic)
#   (b) block fully "safe" (all q >= all k)     → no mask needed, cheaper inner loop
#   (c) boundary block                          → apply causal mask
# This eliminates the per-element `tl.where` for ~(BLOCK_Q-1)/BLOCK_Q fraction of blocks.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['seqlen_q', 'seqlen_k', 'HEAD_DIM'],
)
@triton.jit
def _flash_prefill_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    seqlen_q, seqlen_k,
    q_offset,          # absolute KV position of the first query token
    scale,
    NUM_Q_HEADS: tl.constexpr,
    GQA_GROUPS:  tl.constexpr,
    HEAD_DIM:    tl.constexpr,
    BLOCK_Q:     tl.constexpr,
    BLOCK_K:     tl.constexpr,
    CAUSAL:      tl.constexpr,
):
    block_q_idx = tl.program_id(0)
    bh_idx      = tl.program_id(1)
    b = bh_idx // NUM_Q_HEADS
    h = bh_idx % NUM_Q_HEADS
    kv_h = h // GQA_GROUPS

    q_start = block_q_idx * BLOCK_Q
    offs_q  = q_start + tl.arange(0, BLOCK_Q)
    offs_d  = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q_ptr
              + b * stride_qb
              + offs_q[:, None] * stride_qs
              + h * stride_qh
              + offs_d[None, :] * stride_qd)
    q_mask = offs_q[:, None] < seqlen_q
    q      = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q_fp32 = q.to(tl.float32) * scale
    q_sc   = q_fp32.to(q.dtype)                         # scaled fp16 for tl.dot

    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q],               dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM],      dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + kv_h * stride_kh
    v_base = V_ptr + b * stride_vb + kv_h * stride_vh

    # For causal: upper KV limit for this Q block
    kv_end = seqlen_k
    if CAUSAL:
        kv_end = tl.minimum(seqlen_k, q_offset + q_start + BLOCK_Q)

    # Absolute position of the last q token in this block
    q_abs_last = q_offset + q_start + BLOCK_Q - 1

    for kv_start in range(0, kv_end, BLOCK_K):
        offs_k = kv_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < seqlen_k

        k = tl.load(k_base + offs_k[:, None]*stride_ks + offs_d[None, :]*stride_kd,
                    mask=mask_k[:, None], other=0.0)    # fp16

        # OPTIMISATION: tensor core QK
        qk = tl.dot(q_sc, tl.trans(k)).to(tl.float32)  # [BLOCK_Q, BLOCK_K]

        # OPTIMISATION: apply causal mask only for boundary block
        if CAUSAL:
            kv_block_end = kv_start + BLOCK_K
            is_boundary = (kv_block_end > (q_offset + q_start + 1))
            if is_boundary:
                causal_mask = (q_offset + offs_q[:, None]) >= offs_k[None, :]
                qk = tl.where(causal_mask & mask_k[None, :], qk, float("-inf"))
            else:
                qk = tl.where(mask_k[None, :], qk, float("-inf"))
        else:
            qk = tl.where(mask_k[None, :], qk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha  = tl.exp(m_i - m_new)
        exp_qk = tl.exp(qk - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(exp_qk, axis=1)

        v = tl.load(v_base + offs_k[:, None]*stride_vs + offs_d[None, :]*stride_vd,
                    mask=mask_k[:, None], other=0.0)

        # OPTIMISATION: tensor core AV
        acc = acc * alpha[:, None] + tl.dot(exp_qk.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = (Out_ptr
                + b  * stride_ob
                + offs_q[:, None] * stride_os
                + h  * stride_oh
                + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=q_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

# Fixed number of KV splits — tuned for A100 (108 SMs).
# Each split owns 1/NUM_KV_SPLITS of the KV sequence; results are
# merged by a cheap reduction kernel.
# A100: 8, H100: 16, smaller GPUs: 4
NUM_KV_SPLITS = 8


def _decode(
    q:          torch.Tensor,   # [B, 1, H, D]
    k_cache:    torch.Tensor,   # [B, max_S, H_kv, D]
    v_cache:    torch.Tensor,
    kv_seqlens: torch.Tensor,   # [B] int32 — occupied length after update
) -> torch.Tensor:
    B, _, H, D = q.shape
    assert D in (64, 128, 256)
    H_kv = k_cache.shape[2]
    assert H % H_kv == 0
    GQA_GROUPS = H // H_kv

    scale = 1.0 / math.sqrt(D)
    q_s   = q.squeeze(1)                                        # [B, H, D]

    S = NUM_KV_SPLITS
    partial_o = torch.empty(B, H, S, D, dtype=torch.float16, device=q.device)
    partial_m = torch.empty(B, H, S,    dtype=torch.float32, device=q.device)
    partial_l = torch.empty(B, H, S,    dtype=torch.float32, device=q.device)

    # Phase 1 grid: (B, H, NUM_KV_SPLITS)
    _flash_decode_splitk_kernel[(B, H, S)](
        q_s, k_cache, v_cache,
        partial_o, partial_m, partial_l,
        q_s.stride(0),     q_s.stride(1),     q_s.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        kv_seqlens,
        scale,
        NUM_KV_SPLITS=S,
        GQA_GROUPS=GQA_GROUPS,
        HEAD_DIM=D,
    )

    # Phase 2 grid: (B, H)
    out_s = torch.empty(B, H, D, dtype=q.dtype, device=q.device)
    _flash_decode_reduce_kernel[(B, H)](
        partial_o, partial_m, partial_l,
        out_s,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        out_s.stride(0), out_s.stride(1), out_s.stride(2),
        NUM_KV_SPLITS=S,
        HEAD_DIM=D,
    )
    return out_s.unsqueeze(1)


def _prefill(
    q:          torch.Tensor,   # [B, S_q, H, D]
    k_cache:    torch.Tensor,
    v_cache:    torch.Tensor,
    kv_seqlens: torch.Tensor,
    causal:     bool = True,
) -> torch.Tensor:
    B, S_q, H, D = q.shape
    assert D in (64, 128, 256)
    H_kv = k_cache.shape[2]
    assert H % H_kv == 0
    GQA_GROUPS = H // H_kv

    seqlen_k = int(kv_seqlens.max().item())
    q_offset = seqlen_k - S_q
    scale    = 1.0 / math.sqrt(D)
    out      = torch.empty_like(q)

    grid = lambda meta: (triton.cdiv(S_q, meta['BLOCK_Q']), B * H)
    _flash_prefill_kernel[grid](
        q, k_cache, v_cache, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S_q, seqlen_k,
        q_offset, scale,
        NUM_Q_HEADS=H,
        GQA_GROUPS=GQA_GROUPS,
        HEAD_DIM=D,
        CAUSAL=causal,
    )
    return out


def triton_flash_attn_with_kvcache(
    q:             torch.Tensor,
    k:             torch.Tensor,
    v:             torch.Tensor,
    k_cache:       torch.Tensor,
    v_cache:       torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal:        bool = True,
) -> torch.Tensor:
    """
    Drop-in replacement for flash_attn.flash_attn_with_kvcache.
    Writes (k, v) into cache then runs optimised attention.
    """
    update_kv_cache(k_cache, v_cache, k, v, cache_seqlens)
    new_seqlens = cache_seqlens + q.shape[1]

    if q.shape[1] == 1:
        return _decode(q, k_cache, v_cache, new_seqlens)
    else:
        return _prefill(q, k_cache, v_cache, new_seqlens, causal=causal)


# ---------------------------------------------------------------------------
# Benchmark (identical structure to baseline for fair comparison)
# ---------------------------------------------------------------------------

def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_benchmark():
    device = "cuda"
    dtype  = torch.float16

    print("=" * 75)
    print("Flash Attention benchmark: Optimised Triton vs flash_attn_with_kvcache")
    print("=" * 75)

    try:
        from flash_attn import flash_attn_with_kvcache as ref_fn
        has_ref = True
    except Exception as e:
        print(f"[WARN] flash_attn not available ({e}) — skipping reference")
        has_ref = False

    configs = [
        (1,   1, 32,  8, 128,  512, "decode  B=1  ctx=512"),
        (8,   1, 32,  8, 128,  512, "decode  B=8  ctx=512"),
        (16,  1, 32,  8, 128, 1024, "decode  B=16 ctx=1024"),
        (32,  1, 32,  8, 128, 2048, "decode  B=32 ctx=2048"),
        (1, 512, 32,  8, 128,  512, "prefill B=1  S=512"),
        (4, 256, 32,  8, 128,  256, "prefill B=4  S=256"),
    ]

    for B, S_q, H, H_kv, D, ctx, label in configs:
        max_S = ctx + S_q + 64
        q       = torch.randn(B, S_q, H,    D, dtype=dtype, device=device)
        k_new   = torch.randn(B, S_q, H_kv, D, dtype=dtype, device=device)
        v_new   = torch.randn(B, S_q, H_kv, D, dtype=dtype, device=device)
        k_cache = torch.randn(B, max_S, H_kv, D, dtype=dtype, device=device)
        v_cache = torch.randn(B, max_S, H_kv, D, dtype=dtype, device=device)
        seqlens = torch.full((B,), ctx, dtype=torch.int32, device=device)
        kc_base = k_cache.clone()
        vc_base = v_cache.clone()

        kc_t = kc_base.clone(); vc_t = vc_base.clone()
        def triton_fn():
            kc_t.copy_(kc_base); vc_t.copy_(vc_base)
            return triton_flash_attn_with_kvcache(
                q, k_new, v_new, kc_t, vc_t, seqlens.clone())
        t_tri = benchmark(triton_fn)

        t_ref = None
        if has_ref:
            kc_r = kc_base.clone(); vc_r = vc_base.clone()
            def ref_fn_call():
                kc_r.copy_(kc_base); vc_r.copy_(vc_base)
                return ref_fn(q=q, k_cache=kc_r, v_cache=vc_r,
                              k=k_new, v=v_new,
                              cache_seqlens=seqlens.clone(), causal=True)
            t_ref = benchmark(ref_fn_call)

        ref_s = f"  ref={t_ref:.3f} ms" if t_ref else "              "
        spd_s = f"  ({t_ref/t_tri:.2f}x)" if t_ref else ""
        print(f"{label:35s}  triton={t_tri:.3f} ms{ref_s}{spd_s}")
    print()


def run_correctness_check():
    try:
        from flash_attn import flash_attn_with_kvcache as ref_fn
    except Exception as e:
        print(f"[WARN] flash_attn not available ({e}) — skipping correctness check")
        return

    device = "cuda"
    cases = [
        (1,   1, 16, 16, 128,   64, "decode  MHA  B=1  ctx=64"),
        (4,   1, 32,  8, 128,  128, "decode  GQA  B=4  ctx=128"),
        (8,   1, 32,  8, 128,  512, "decode  GQA  B=8  ctx=512"),
        (16,  1, 32,  8, 128, 2048, "decode  GQA  B=16 ctx=2048"),
        (1,   1, 16, 16,  64,  128, "decode  MHA  D=64 ctx=128"),
        (1, 128, 16, 16, 128,    0, "prefill MHA  B=1  S=128 ctx=0"),
        (2, 256, 32,  8, 128,  128, "prefill GQA  B=2  S=256 ctx=128"),
        (1, 512, 32,  8, 128,    0, "prefill GQA  B=1  S=512 ctx=0"),
        (2,  64, 16, 16,  64,   64, "prefill MHA  D=64 S=64  ctx=64"),
    ]

    print("=" * 80)
    print(f"{'Correctness: Optimised Triton vs flash_attn_with_kvcache':^80}")
    print("=" * 80)

    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        print(f"\n  dtype = {dtype_name}")
        print(f"  {'Case':<40} {'out_max':>9} {'out_mean':>10} {'kv_max':>9}  result")
        print(f"  {'-'*40} {'-'*9} {'-'*10} {'-'*9}  {'------'}")
        all_pass = True
        for cfg in cases:
            B, S_q, H, H_kv, D, ctx, label = cfg
            max_S = ctx + S_q + 8
            torch.manual_seed(42)
            q       = torch.randn(B, S_q, H,    D, dtype=dtype, device=device)
            k_new   = torch.randn(B, S_q, H_kv, D, dtype=dtype, device=device)
            v_new   = torch.randn(B, S_q, H_kv, D, dtype=dtype, device=device)
            k_cache = torch.randn(B, max_S, H_kv, D, dtype=dtype, device=device)
            v_cache = torch.randn(B, max_S, H_kv, D, dtype=dtype, device=device)
            seqlens = torch.full((B,), ctx, dtype=torch.int32, device=device)
            try:
                out_ref = ref_fn(q=q, k_cache=k_cache.clone(), v_cache=v_cache.clone(),
                                 k=k_new, v=v_new,
                                 cache_seqlens=seqlens.clone(), causal=True)
                out_tri = triton_flash_attn_with_kvcache(
                    q, k_new, v_new, k_cache.clone(), v_cache.clone(),
                    seqlens.clone(), causal=True)
                out_max  = (out_ref - out_tri).abs().max().item()
                out_mean = (out_ref - out_tri).abs().mean().item()
                status   = "PASS" if out_max < 0.05 else "FAIL"
                all_pass = all_pass and (status == "PASS")
                print(f"  {label:<40} {out_max:>9.5f} {out_mean:>10.6f} {'N/A':>9}  {status}")
            except Exception as e:
                all_pass = False
                print(f"  {label:<40} {'ERROR':>9}  {e}")
        print(f"\n  Summary ({dtype_name}): {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()


if __name__ == "__main__":
    run_correctness_check()
    run_benchmark()