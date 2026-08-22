"""Qwen3 paged GQA decode attention (TileLang) — v4 每 q-head 一个 block（仅 row0）。

grid=(bs, num_heads)=16 block（bs=1）。每 block 处理 1 个 q-head，遍历该 q-head 对应
kv_head 的全部 KV。GQA: q_head h → kv_head h//q_per_kv。2 个 q-head 共享 1 kv-head，
KV 被读 2 次（开销可接受，KV 小且开销主导）。

每 block 仅用 row0（fragment[0,...] 合法，无非零索引），M=16 pad 供 mma。
chunk 级 online softmax（flash 算法），row0 唯一有效行。
K 已旋转（cache 存旋转 K），仅旋转 Q。
"""
import torch
import tilelang
import tilelang.language as T

_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def qwen3_decode_attn_kernel(
    bs, num_heads, kv_num_heads, head_size, block_size, max_seq_blocks,
    n_blocks, max_pos, dtype,
):
    accum = T.float32
    half = head_size // 2
    q_per_kv = num_heads // kv_num_heads  # 2
    scale = (1.0 / (head_size ** 0.5))
    BLOCK_K = 64

    @T.prim_func
    def main(
        QKV: T.Tensor([bs, num_heads * head_size + 2 * kv_num_heads * head_size], dtype),
        KCache: T.Tensor([n_blocks, block_size, kv_num_heads, head_size], dtype),
        VCache: T.Tensor([n_blocks, block_size, kv_num_heads, head_size], dtype),
        BlockTable: T.Tensor([bs, max_seq_blocks], T.int32),
        CacheSeqLens: T.Tensor([bs], T.int32),
        Cos: T.Tensor([max_pos, half], dtype),
        Sin: T.Tensor([max_pos, half], dtype),
        Out: T.Tensor([bs, num_heads * head_size], dtype),
    ):
        with T.Kernel(bs * num_heads, threads=32) as (bh,):
            b = bh // num_heads
            qh = bh % num_heads
            kvh = qh // q_per_kv
            seqlen = CacheSeqLens[b]
            qpos = seqlen

            # ---- 旋转 Q，pad [16, head_size]（row0 真实）----
            Qpad = T.alloc_shared([16, head_size], dtype)
            T.clear(Qpad)
            cs = T.alloc_shared([half], dtype)
            ss = T.alloc_shared([half], dtype)
            for j in T.Parallel(half):
                cs[j] = Cos[qpos, j]; ss[j] = Sin[qpos, j]
            qa = T.alloc_fragment([half], accum)
            qb = T.alloc_fragment([half], accum)
            for j in T.Parallel(half):
                qa[j] = T.cast(QKV[b, qh * head_size + j], accum)
                qb[j] = T.cast(QKV[b, qh * head_size + j + half], accum)
            for j in T.Parallel(half):
                Qpad[0, j] = T.cast(qa[j] * T.cast(cs[j], accum) - qb[j] * T.cast(ss[j], accum), dtype)
                Qpad[0, j + half] = T.cast(qb[j] * T.cast(cs[j], accum) + qa[j] * T.cast(ss[j], accum), dtype)

            # ---- 存新 K/V 到 paged cache（position seqlen）----
            # 每 kv_head 仅 qh%q_per_kv==0 的 block 存（避免 2 个 q-head 重复写同一 kv 位置）。
            # K 旋转后存（与 flash 一致：cache 存旋转 K）；V 原值存。
            if qh % q_per_kv == 0:
                k_blk = seqlen // block_size
                k_off = seqlen % block_size
                k_bid = BlockTable[b, k_blk]
                q_dim = num_heads * head_size
                kv_dim = kv_num_heads * head_size
                # K: qkv[b, q_dim + kvh*head_size + j]，旋转 half-split 后存
                ka = T.alloc_fragment([half], accum)
                kb = T.alloc_fragment([half], accum)
                for j in T.Parallel(half):
                    ka[j] = T.cast(QKV[b, q_dim + kvh * head_size + j], accum)
                    kb[j] = T.cast(QKV[b, q_dim + kvh * head_size + j + half], accum)
                for j in T.Parallel(half):
                    KCache[k_bid, k_off, kvh, j] = T.cast(ka[j] * T.cast(cs[j], accum) - kb[j] * T.cast(ss[j], accum), dtype)
                    KCache[k_bid, k_off, kvh, j + half] = T.cast(kb[j] * T.cast(cs[j], accum) + ka[j] * T.cast(ss[j], accum), dtype)
                # V: qkv[b, q_dim + kv_dim + kvh*head_size + j]，原值存
                for j in T.Parallel(head_size):
                    VCache[k_bid, k_off, kvh, j] = QKV[b, q_dim + kv_dim + kvh * head_size + j]

            # ---- online softmax 状态（标量 fragment[0]）----
            m = T.alloc_fragment([1], accum)
            s = T.alloc_fragment([1], accum)
            acc = T.alloc_fragment([16, head_size], accum)
            T.fill(m, -1e30); T.clear(s); T.clear(acc)

            Ks = T.alloc_shared([head_size, BLOCK_K], dtype)
            Vs = T.alloc_shared([BLOCK_K, head_size], dtype)
            qk = T.alloc_fragment([16, BLOCK_K], accum)
            qk_s = T.alloc_shared([16, BLOCK_K], dtype)   # gemm 后落 shared 供标量读
            p = T.alloc_shared([16, BLOCK_K], dtype)
            pv = T.alloc_fragment([16, head_size], accum)
            pv_s = T.alloc_shared([16, head_size], dtype)

            # attention 覆盖 0..seqlen（含刚存的新 K at position seqlen）= seqlen+1 个位置。
            # flash 行为：decode token at position seqlen 因果关注 0..seqlen（含自身新 K）。
            atten = seqlen + 1
            n_blk = T.ceildiv(atten, block_size)
            for blk in T.serial(n_blk):
                bid = BlockTable[b, blk]
                blk_start = blk * block_size
                n_in_blk = T.min(block_size, atten - blk_start)
                n_chunk = T.ceildiv(n_in_blk, BLOCK_K)
                for ci in T.serial(n_chunk):
                    k_start = ci * BLOCK_K
                    # K 分块（转置）KCache[bid, off, kvh, d] → Ks[d, c]
                    for c, d in T.Parallel(BLOCK_K, head_size):
                        off = k_start + c
                        Ks[d, c] = T.if_then_else(off < n_in_blk, KCache[bid, off, kvh, d], T.cast(0, dtype))
                    T.clear(qk)
                    T.gemm(Qpad, Ks, qk, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(qk, qk_s)
                    # rowmax（仅 row0，串行 reduce over BLOCK_K）
                    mc = T.alloc_fragment([1], accum)
                    T.fill(mc, -1e30)
                    for c in T.serial(BLOCK_K):
                        valid = (k_start + c) < n_in_blk
                        v0 = T.if_then_else(valid, qk_s[0, c] * scale, -1e30)
                        mc[0] = T.max(mc[0], v0)
                    m_new = T.max(m[0], mc[0])
                    f = T.exp(m[0] - m_new)
                    # p = exp(qk*scale - m_new)，rowsum
                    sc = T.alloc_fragment([1], accum)
                    T.clear(sc)
                    for c in T.serial(BLOCK_K):
                        valid = (k_start + c) < n_in_blk
                        e0 = T.if_then_else(valid, T.exp(qk_s[0, c] * scale - m_new), T.cast(0, accum))
                        p[0, c] = T.cast(e0, dtype)
                        sc[0] = sc[0] + e0
                    # rescale acc *= f（row0）
                    for d in T.Parallel(head_size):
                        acc[0, d] = acc[0, d] * f
                    # V 分块 VCache[bid, off, kvh, d] → Vs[c, d]
                    for c, d in T.Parallel(BLOCK_K, head_size):
                        off = k_start + c
                        Vs[c, d] = T.if_then_else(off < n_in_blk, VCache[bid, off, kvh, d], T.cast(0, dtype))
                    T.clear(pv)
                    T.gemm(p, Vs, pv, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(pv, pv_s)
                    for d in T.Parallel(head_size):
                        acc[0, d] = acc[0, d] + pv_s[0, d]
                    s[0] = s[0] * f + sc[0]
                    m[0] = m_new

            inv_s = 1.0 / s[0]
            for d in T.Parallel(head_size):
                Out[b, qh * head_size + d] = T.cast(acc[0, d] * inv_s, dtype)
    return main


_cache = {}


def get_decode_attn(bs, num_heads, kv_num_heads, head_size, block_size, max_seq_blocks,
                    n_blocks, max_pos, dtype):
    key = (bs, num_heads, kv_num_heads, head_size, block_size, max_seq_blocks, n_blocks, max_pos, dtype)
    if key not in _cache:
        _cache[key] = qwen3_decode_attn_kernel(
            bs, num_heads, kv_num_heads, head_size, block_size, max_seq_blocks,
            n_blocks, max_pos, _TORCH_TO_TL[dtype])
    return _cache[key]


def qwen3_decode_attn(qkv, k_cache, v_cache, block_table, cache_seqlens,
                      cos, sin, num_heads, kv_num_heads, head_size, out=None):
    """qkv: [bs, q_dim+2*kv_dim]；out: [bs, num_heads*head_size] 预分配 buffer（graph 友好）。
    返回 out（或新分配）。
    """
    bs = qkv.shape[0]
    n_blocks, block_size, _, _ = k_cache.shape
    max_seq_blocks = block_table.shape[1]
    max_pos = cos.shape[0]
    dtype = qkv.dtype
    kernel = get_decode_attn(bs, num_heads, kv_num_heads, head_size, block_size,
                             max_seq_blocks, n_blocks, max_pos, dtype)
    if out is None:
        out = torch.empty(bs, num_heads * head_size, dtype=dtype, device=qkv.device)
    kernel(qkv, k_cache, v_cache, block_table, cache_seqlens, cos, sin, out)
    return out
