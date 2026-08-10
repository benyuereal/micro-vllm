#!/usr/bin/env python3
"""bs=1: 进一步拆 cat+flash+oproj 与 routed MoE 的 gate_up vs down。"""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.adapter import rmsnorm
from flash_attn import flash_attn_varlen_func
from kernel.grouped_gemv import grouped_gate_up, grouped_down


def bench_graph(fn_build, n_iter=300):
    for _ in range(5): fn_build()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn_build()
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


def main():
    print("Loading ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
    engine.add_request("写SpringBoot文件上传代码", max_tokens=150, temperature=0.0)
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

    # 重建 latents / kv / q_full / k_full / v_fa 给 cat/flash/oproj 拆分用
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
    k_v = k_full.reshape(bs * max_len, num_heads, q_head)
    v_v = v_fa.reshape(bs * max_len, num_heads, q_head)

    def step_flash():
        return flash_attn_varlen_func(q_full, k_v, v_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=max_len, softmax_scale=gr._ds_softmax_scale, causal=False)

    ao = step_flash()
    def step_oproj():
        o = ao[..., :v_head].reshape(bs, num_heads * v_head)
        return F.linear(o, attn._o_w, attn._o_b)

    for _ in range(3): step_flash(); step_oproj()
    torch.cuda.synchronize()
    print(f"\n=== bs=1 cat+flash+oproj 细拆 ===", flush=True)
    t_flash = bench_graph(step_flash)
    print(f"  flash_attn_varlen:  {t_flash:6.1f} us")
    t_oproj = bench_graph(step_oproj)
    print(f"  o_proj:             {t_oproj:6.1f} us")
    print(f"  (cat ≈ {74.2 - t_flash - t_oproj:.1f} us, 整段 74.2)")

    # ---- routed MoE: gate_up vs silu*up*w vs down ----
    mlp = moe_block.mlp
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    x_in = gr._h_buf[:bs]
    logits = F.linear(x_in, mlp._gate_w)
    scores = logits.softmax(dim=-1, dtype=torch.float32).to(x_in.dtype)
    topk_w, topk_idx = torch.topk(scores, k=A._top_k, dim=-1, sorted=False)
    idx_i = topk_idx.reshape(-1).to(torch.int64)
    w_i = topk_w.reshape(-1)
    w_ones = torch.ones(A._top_k, dtype=x_in.dtype, device=x_in.device)

    def step_gu():
        return grouped_gate_up(x_in, mlp._e_gu, idx_i)
    gu = step_gu()
    gate, up = gu.chunk(2, dim=-1)
    def step_silu():
        return F.silu(gate) * up * w_i.unsqueeze(-1).to(gu.dtype)
    act = step_silu()
    def step_down():
        return grouped_down(act, mlp._e_d, idx_i, w_ones)

    for _ in range(3): step_gu(); step_silu(); step_down()
    torch.cuda.synchronize()
    print(f"\n=== bs=1 routed MoE 细拆 (6 expert) ===", flush=True)
    t_gu = bench_graph(step_gu)
    print(f"  grouped_gate_up (6×[1,2048]@[2048,2816]): {t_gu:6.1f} us")
    t_silu = bench_graph(step_silu)
    print(f"  silu*up*w:                                {t_silu:6.1f} us")
    t_down = bench_graph(step_down)
    print(f"  grouped_down (6×[1,1408]@[1408,2048]):    {t_down:6.1f} us")
    print(f"  (整段 routed 117.2, 合计 {t_gu+t_silu+t_down:.1f})")


if __name__ == "__main__":
    main()
