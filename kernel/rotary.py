"""Decode 专用融合 RoPE + slot_mapping 计算。

替代 flash_attn_with_kvcache 的 internal-rotary 路径：在 flash 前显式旋转 q/k 并
store k/v，flash 跑纯 attention（无 rotary_cos/sin、无 k=/v=）。

profiled（bs=512, 28 层）：internal-rotary+k=/v= 298.6us/层 → prerope+store+pure-flash
247.8us/层，省 50.8us/层 × 28 = 1.42ms/step，是 micro 落后 nano 的最大单项 gap。

RoPE half-split：q/k[..., :d/2] 与 [d/2:]，cos/sin 表 [max_pos, d/2]，按 per-seq 位置 gather。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_decode_kernel(QK, COS, SIN, POS,
                          stride_b, stride_h,
                          dim: tl.constexpr, half: tl.constexpr,
                          BLOCK: tl.constexpr):
    """in-place RoPE on q or k: [bs, heads, dim]。一个 program = (batch, head)。
    位置由 POS[batch]（cache_seqlens）给出。half-split 旋转。"""
    b = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + b)
    base = b * stride_b + h * dim
    offs = tl.arange(0, BLOCK)
    mask = offs < half
    c = tl.load(COS + pos * half + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(SIN + pos * half + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(QK + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(QK + base + half + offs, mask=mask, other=0.0).to(tl.float32)
    o1 = (x1 * c - x2 * s).to(QK.dtype.element_ty)
    o2 = (x2 * c + x1 * s).to(QK.dtype.element_ty)
    tl.store(QK + base + offs, o1, mask=mask)
    tl.store(QK + base + half + offs, o2, mask=mask)


def apply_rope_decode(t: torch.Tensor, cos_pool: torch.Tensor, sin_pool: torch.Tensor,
                      positions: torch.Tensor):
    """in-place half-split RoPE on [bs, heads, dim]。positions=[bs] int32 (cache_seqlens)。"""
    bs, nh, d = t.shape
    half = d // 2
    _rotary_decode_kernel[(bs, nh)](
        t, cos_pool, sin_pool, positions,
        t.stride(0), t.stride(1),
        dim=d, half=half, BLOCK=triton.next_power_of_2(half))


@triton.jit
def _slot_mapping_kernel(BLOCK_TABLE, SEQLENS, OUT, BLOCK_SIZE: tl.constexpr,
                         MAX_SEQ_BLOCKS: tl.constexpr):
    """slot_mapping[i] = block_table[i, seqlen_i // block_size] * block_size
                          + seqlen_i % block_size。一个 program = 一个 seq。"""
    i = tl.program_id(0)
    sl = tl.load(SEQLENS + i)
    blk_idx = sl // BLOCK_SIZE
    # block_table 列数足够（启动期 assert 保证 seqlen < max_seq_blocks*block_size）
    block_id = tl.load(BLOCK_TABLE + i * MAX_SEQ_BLOCKS + blk_idx)
    slot = block_id * BLOCK_SIZE + (sl % BLOCK_SIZE)
    tl.store(OUT + i, slot)


def compute_slot_mapping(block_table: torch.Tensor, cache_seqlens: torch.Tensor,
                         block_size: int, out: torch.Tensor):
    """算当前步各 seq 的写入 slot。block_table [bs, max_seq_blocks] int32，
    cache_seqlens [bs] int32（+1 前），out [bs] int32 预分配。"""
    bs = cache_seqlens.shape[0]
    max_seq_blocks = block_table.shape[1]
    _slot_mapping_kernel[(bs,)](
        block_table, cache_seqlens, out,
        BLOCK_SIZE=block_size, MAX_SEQ_BLOCKS=max_seq_blocks)
    return out


# ---- QK-Norm + RoPE 融合（Qwen3 decode prerope 路径专用）----
# 替代分离的 qk_norm_inplace + apply_rope_decode 两 kernel：单 kernel 读 head_dim → 算 RMSNorm →
# half-split RoPE → 写回，省中间一次 head_dim 读+写（profiled 省 ~100us/step @ bs=512）。
@triton.jit
def _qk_norm_rope_kernel(QKV, W, COS, SIN, POS,
                         stride_qkv_row, seg_offset, head_size: tl.constexpr,
                         num_heads: tl.constexpr, half: tl.constexpr,
                         eps, BLOCK_H: tl.constexpr, BLOCK_HALF: tl.constexpr):
    """每个 program 处理一个 (batch, head)。
    pid = batch_idx * num_heads + head_idx
    head 在 qkv_buf 中的起始 = batch_idx*stride_qkv_row + seg_offset + head_idx*head_size
    两遍：第一遍算 mean_sq → rrms；第二遍 norm×weight + half-split RoPE 原地写回。
    half-split RoPE: out[:half] = xn[:half]*c - xn[half:]*s; out[half:] = xn[half:]*c + xn[:half]*s"""
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    base = batch_idx * stride_qkv_row + seg_offset + head_idx * head_size
    pos = tl.load(POS + batch_idx)

    # 第一遍：读全 head_dim 算 mean_sq → rrms
    offs = tl.arange(0, BLOCK_H)
    mask = offs < head_size
    x = tl.load(QKV + base + offs, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / head_size + eps)

    # 第二遍：分别读前 half / 后 half，norm×weight + half-split RoPE 后写回
    h_offs = tl.arange(0, BLOCK_HALF)
    h_mask = h_offs < half
    c = tl.load(COS + pos * half + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    s = tl.load(SIN + pos * half + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W + half + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    x1 = tl.load(QKV + base + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    x2 = tl.load(QKV + base + half + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    xn1 = x1 * rrms * w1
    xn2 = x2 * rrms * w2
    o1 = (xn1 * c - xn2 * s).to(QKV.dtype.element_ty)
    o2 = (xn2 * c + xn1 * s).to(QKV.dtype.element_ty)
    tl.store(QKV + base + h_offs, o1, mask=h_mask)
    tl.store(QKV + base + half + h_offs, o2, mask=h_mask)


def qk_norm_rope_inplace(qkv_buf, bs, seg_offset, num_heads, head_size,
                         norm_weight, cos_pool, sin_pool, positions, eps=1e-6):
    """对 qkv_buf 的某段（q 段 seg_offset=0 或 k 段 seg_offset=q_dim）原地做
    per-head QK-Norm + half-split RoPE 融合。positions=[bs] int32 (cache_seqlens)。"""
    half = head_size // 2
    BLOCK_H = triton.next_power_of_2(head_size)
    BLOCK_HALF = triton.next_power_of_2(half)
    _qk_norm_rope_kernel[(bs * num_heads,)](
        qkv_buf, norm_weight, cos_pool, sin_pool, positions,
        qkv_buf.stride(0), seg_offset, head_size, num_heads, half,
        eps, BLOCK_H=BLOCK_H, BLOCK_HALF=BLOCK_HALF)


