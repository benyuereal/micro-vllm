#!/usr/bin/env python3
"""长上下文正确性：把 decode 推到 ~900 tokens，让 4 个 split 都有数据，验证 combine 路径。"""
import sys, time, torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.adapter import rmsnorm
from kernel.tilelang_mla import _get_kernel
import torch.nn.functional as F


def run_to_len(engine, target_len):
    """decode 直到序列长度 >= target_len，返回到达时的 ctx。"""
    n = 0
    while True:
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        n += 1
        if bt == "prefill":
            continue
        # decode step done; check current seq len
        cur = engine.cache_manager._cache_seqlens_buffer[0].item()
        if cur >= target_len:
            return cur


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
    engine.add_request("请详细解释 Transformer 架构中多头自注意力机制的完整计算流程，包括 Q K V 矩阵的生成、缩放点积注意力、softmax 归一化、多头拼接和输出投影，并分析为什么需要位置编码以及 RoPE 旋转位置编码相比绝对位置编码的优势。", max_tokens=900, temperature=0.0)
    # prefill
    while True:
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b: time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break
    # decode to ~900
    cur = 0
    for _ in range(1000):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "decode":
            cur = engine.cache_manager._cache_seqlens_buffer[0].item()
            if cur >= 900: break
    print(f"reached context len = {cur}", flush=True)
    if cur < 200:
        print("context too short, abort"); return

    gr = engine.graph_runner; A = gr.adapter
    blocks = A.blocks(gr.model); H = A._hidden
    cm = engine.cache_manager; bt_buf = cm._block_table_buffer
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen
    moe_block = next(b for b in blocks if getattr(b.mlp, "_is_moe", False))
    attn = moe_block.self_attn

    bs = 1
    layer_idx = [i for i, b in enumerate(blocks) if getattr(b.mlp, "_is_moe", False)][0]
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    A.compute_qkv(moe_block, gr._h_buf[:bs], gr, bs)
    torch.cuda.synchronize()

    q = attn._q_cache
    k_cache, v_cache = cm.get(layer_idx)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    new_pos = (cache_lens - 1).long().clamp(min=0)
    max_len = gr._cur_bucket_maxlen
    block_size = cm.block_size
    kv_lora = A._kv_lora_rank; qk_rope = A._qk_rope
    qk_nope = A._qk_nope; v_head = A._v_head; num_heads = A._num_heads
    q_head = A._q_head; latent_dim = A._latent_dim

    compressed_kv_new = attn._compressed_kv
    k_pe_new = attn._k_pe
    latent_new = torch.cat([compressed_kv_new, k_pe_new], dim=-1).view(bs, 1, 1, latent_dim)
    slots = A._decode_slots(bt_buf, new_pos, bs, block_size)
    A._store_latent_batch(latent_new, k_cache, v_cache, slots, block_size)

    total_lens = cache_lens.long()
    bt = bt_buf[:bs].long()
    t_idx = torch.arange(max_len, device=bt.device)
    blk_id = bt[:, t_idx // block_size].clamp(min=0)
    n_slots = k_cache.shape[0] * block_size
    gslots = (blk_id * block_size + (t_idx % block_size)).clamp(min=0, max=n_slots - 1)
    k_flat = k_cache.reshape(-1, latent_dim)
    latents = k_flat[gslots.reshape(-1)].view(bs, max_len, latent_dim)
    compressed_kv, k_pe_all = latents.split([kv_lora, qk_rope], dim=-1)
    ckv = rmsnorm(compressed_kv.reshape(-1, kv_lora), attn._kva_ln_w, attn._kva_ln_eps)
    kv = F.linear(ckv, attn._kvb_w).view(bs, max_len, num_heads, qk_nope + v_head)
    k_nope, v = kv.split([qk_nope, v_head], dim=-1)
    cos, sin = A._rope_pool(gr, k_cache.device)
    q_nope, q_pe = q.split([qk_nope, qk_rope], dim=-1)
    cos_q = cos[new_pos].unsqueeze(1); sin_q = sin[new_pos].unsqueeze(1)
    q_pe_r = A._apply_rope(q_pe, cos_q, sin_q)
    k_pos = torch.arange(max_len, device=k_cache.device).unsqueeze(0)
    k_pe_rot = A._apply_rope(k_pe_all, cos[k_pos], sin[k_pos])
    q_full = torch.cat([q_nope, q_pe_r], dim=-1)
    k_full = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, num_heads, -1)], dim=-1)
    v_fa = F.pad(v, (0, q_head - v_head))
    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q_full.device)
    cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q_full.device)
    cu_k[1:] = torch.cumsum(total_lens.to(torch.int32), dim=0)
    from flash_attn import flash_attn_varlen_func
    attn_out_ref = flash_attn_varlen_func(
        q_full, k_full.reshape(bs * max_len, num_heads, q_head), v_fa.reshape(bs * max_len, num_heads, q_head),
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=max_len,
        softmax_scale=gr._ds_softmax_scale, causal=False)
    attn_out_ref = attn_out_ref[..., :v_head]

    kvb_w_full = attn._kvb_w.view(num_heads, qk_nope + v_head, kv_lora)
    kvb_w_kn = kvb_w_full[:, :qk_nope, :].contiguous()
    kvb_w_v = kvb_w_full[:, qk_nope:, :].contiguous()
    A_in = torch.einsum('bhd,hdk->bhk', q_nope.float(), kvb_w_kn.float()).to(engine.dtype).contiguous()
    Q_pe_in = q_pe_r.contiguous()
    Latent_flat = k_cache.reshape(-1, 1, latent_dim).contiguous()
    bt_kernel = bt_buf[:bs].to(torch.int32).contiguous()
    cache_seqlens_in = cache_lens.to(torch.int32).contiguous()
    cos_full, sin_full = A._rope_pool(gr, k_cache.device)
    cos_k_in = cos_full[:max_len].contiguous()
    sin_k_in = sin_full[:max_len].contiguous()

    softmax_scale = gr._ds_softmax_scale
    block_N = 64; num_split = 4
    n_slots = k_cache.shape[0] * block_size
    kernel = _get_kernel(bs, num_heads, max_len, kv_lora, qk_rope, qk_nope, v_head,
                         block_size, softmax_scale, engine.dtype, n_slots,
                         block_N=block_N, num_split=num_split)
    out = kernel(A_in, Q_pe_in, Latent_flat, bt_kernel, cache_seqlens_in,
                 attn._kva_ln_w, kvb_w_v, cos_k_in, sin_k_in)
    diff = (out.float() - attn_out_ref.float()).abs()
    rel = diff.max().item() / attn_out_ref.float().abs().max().item()
    print(f"[ctx={cur}] max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.4f} rel_max={rel:.4f} OK={rel < 0.05}")


if __name__ == "__main__":
    main()
