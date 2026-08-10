#!/usr/bin/env python3
"""验证 attention 的 cat/expand/reshape 到底哪一步贵。"""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.adapter import rmsnorm


def bench_graph(fn, n_iter=300):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


def main():
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
    engine.add_request("x", max_tokens=150, temperature=0.0)
    for _ in range(300):
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b: time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break
    for _ in range(3):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    gr = engine.graph_runner; A = gr.adapter
    blocks = A.blocks(gr.model); H = A._hidden; bs = 1
    cm = engine.cache_manager; bt_buf = cm._block_table_buffer
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen
    moe_block = next(b for b in blocks if getattr(b.mlp, "_is_moe", False))
    attn = moe_block.self_attn
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    A.compute_qkv(moe_block, gr._h_buf[:bs], gr, bs)
    torch.cuda.synchronize()

    q = attn._q_cache
    k_cache, v_cache = cm.get(0)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    new_pos = (cache_lens - 1).long().clamp(min=0)
    max_len = gr._cur_bucket_maxlen; block_size = cm.block_size
    latent_dim = A._latent_dim; kv_lora = A._kv_lora_rank; qk_rope = A._qk_rope
    qk_nope = A._qk_nope; v_head = A._v_head; q_head = A._q_head; num_heads = A._num_heads

    # 重建中间量
    total_lens = cache_lens.long()
    bt = bt_buf[:bs].long()
    t_idx = torch.arange(max_len, device=bt.device)
    blk_id = bt[:, t_idx // block_size].clamp(min=0)
    n_slots = k_cache.shape[0] * block_size
    slots = (blk_id * block_size + (t_idx % block_size)).clamp(min=0, max=n_slots - 1)
    latents = k_cache.reshape(-1, latent_dim)[slots.reshape(-1)].view(bs, max_len, latent_dim)
    compressed_kv, k_pe_all = latents.split([kv_lora, qk_rope], dim=-1)
    ckv = rmsnorm(compressed_kv.reshape(-1, kv_lora), attn._kva_ln_w, attn._kva_ln_eps)
    kv = F.linear(ckv, attn._kvb_w).view(bs, max_len, num_heads, qk_nope + v_head)
    k_nope, v = kv.split([qk_nope, v_head], dim=-1)
    cos, sin = A._rope_pool(gr, k_cache.device)
    q_nope, q_pe = q.split([qk_nope, qk_rope], dim=-1)
    q_pe_r = A._apply_rope(q_pe, cos[new_pos].unsqueeze(1), sin[new_pos].unsqueeze(1))
    k_pe_rot = A._apply_rope(k_pe_all, cos[torch.arange(max_len, device=k_cache.device).unsqueeze(0)],
                             sin[torch.arange(max_len, device=k_cache.device).unsqueeze(0)])

    # 各子步
    def s_qcat(): return torch.cat([q_nope, q_pe_r], dim=-1)
    def s_kexpand(): return k_pe_rot.unsqueeze(2).expand(-1, -1, num_heads, -1)
    ke = s_kexpand()
    def s_kcat(): return torch.cat([k_nope, ke], dim=-1)
    def s_vpad(): return F.pad(v, (0, q_head - v_head))
    def s_reshape_k(): return s_kcat().reshape(bs * max_len, num_heads, q_head)
    def s_all():
        qf = torch.cat([q_nope, q_pe_r], dim=-1)
        kf = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, num_heads, -1)], dim=-1)
        vf = F.pad(v, (0, q_head - v_head))
        return qf, kf.reshape(bs * max_len, num_heads, q_head), vf.reshape(bs * max_len, num_heads, q_head)

    for _ in range(3): s_all()
    torch.cuda.synchronize()
    print(f"\n=== bs=1 cat/expand/reshape 细拆 (max_len={max_len}) ===", flush=True)
    print(f"  q cat [1,H,128]+[1,H,64]→[1,H,192]:       {bench_graph(s_qcat):6.1f} us")
    print(f"  k_pe expand [1,L,64]→[1,L,H,64]:           {bench_graph(s_kexpand):6.1f} us")
    print(f"  k cat [1,L,H,128]+[1,L,H,64]→[1,L,H,192]:  {bench_graph(s_kcat):6.1f} us")
    print(f"  v pad [1,L,H,128]→[1,L,H,192]:             {bench_graph(s_vpad):6.1f} us")
    print(f"  k reshape → [L,H,192]:                      {bench_graph(s_reshape_k):6.1f} us")
    print(f"  全部(qcat+kexpand+kcat+vpad+2reshape):      {bench_graph(s_all):6.1f} us")


if __name__ == "__main__":
    main()
