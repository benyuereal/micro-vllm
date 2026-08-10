#!/usr/bin/env python3
"""bs=1 graph 下 attention 内部各子段 + MoE 内部各子段 GPU 时间。

attention 157us 拆成：store / gather / kvb(=rmsnorm+kvb_proj+split) / rope / cat / flash / oproj
MoE 179us 拆成：gate_up GEMV / silu*up / down GEMV / shared / gate_topk
确定 bs=1 单层全融合时哪些子段值得并入 kernel。
"""
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.adapter import DeepSeekAdapter, rmsnorm, rmsnorm_, rmsnorm_residual
from flash_attn import flash_attn_varlen_func


def bench_graph(fn_build, n_iter=300):
    for _ in range(5):
        fn_build()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn_build()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


def main():
    print("Loading engine ...", flush=True)
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

    gr = engine.graph_runner
    A = gr.adapter
    blocks = A.blocks(gr.model)
    H = A._hidden
    bs = 1
    cm = engine.cache_manager
    bt_buf = cm._block_table_buffer
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen
    moe_block = next(b for b in blocks if getattr(b.mlp, "_is_moe", False))
    attn = moe_block.self_attn

    # 准备输入
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    h = gr._h_buf[:bs]
    A.compute_qkv(moe_block, h, gr, bs)  # 设好 _q_cache 等
    torch.cuda.synchronize()

    # ---- attention 子段拆解 ----
    # 预算中间张量（graph 内重算即可，但子段计时需各自连贯）。用独立 buffer。
    q = attn._q_cache
    compressed_kv_new = attn._compressed_kv
    k_pe_new = attn._k_pe
    k_cache, v_cache = cm.get(0)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    new_pos = (cache_lens - 1).long().clamp(min=0)
    max_len = gr._cur_bucket_maxlen
    block_size = cm.block_size
    latent_dim = A._latent_dim
    kv_lora = A._kv_lora_rank
    qk_rope = A._qk_rope
    qk_nope = A._qk_nope
    v_head = A._v_head
    q_head = A._q_head
    num_heads = A._num_heads

    def step_store():
        latent_new = torch.cat([compressed_kv_new, k_pe_new], dim=-1)
        latent_new = latent_new.view(bs, 1, 1, latent_dim)
        slots = A._decode_slots(bt_buf, new_pos, bs, block_size)
        A._store_latent_batch(latent_new, k_cache, v_cache, slots, block_size)

    def step_gather():
        total_lens = cache_lens.long()
        bt = bt_buf[:bs].long()
        t_idx = torch.arange(max_len, device=bt.device)
        blk_idx = t_idx // block_size
        off_idx = t_idx % block_size
        blk_id = bt[:, blk_idx].clamp(min=0)
        n_slots = k_cache.shape[0] * block_size
        slots = (blk_id * block_size + off_idx).clamp(min=0, max=n_slots - 1)
        k_flat = k_cache.reshape(-1, latent_dim)
        return k_flat[slots.reshape(-1)].view(bs, max_len, latent_dim)

    # 预算 gather 结果给后续子段用
    latents = step_gather()

    def step_kvb():
        compressed_kv, k_pe_all = latents.split([kv_lora, qk_rope], dim=-1)
        ckv = rmsnorm(compressed_kv.reshape(-1, kv_lora), attn._kva_ln_w, attn._kva_ln_eps)
        kv = F.linear(ckv, attn._kvb_w).view(bs, max_len, num_heads, qk_nope + v_head)
        return kv

    kv = step_kvb()

    def step_rope():
        cos, sin = A._rope_pool(gr, k_cache.device)
        q_nope, q_pe = q.split([qk_nope, qk_rope], dim=-1)
        cos_q = cos[new_pos].unsqueeze(1); sin_q = sin[new_pos].unsqueeze(1)
        q_pe_r = A._apply_rope(q_pe, cos_q, sin_q)
        k_pos = torch.arange(max_len, device=k_cache.device).unsqueeze(0)
        cos_k = cos[k_pos]; sin_k = sin[k_pos]
        k_pe_rot = A._apply_rope(k_pe_all_buf[0], cos_k, sin_k)
        return q_nope, q_pe_r, k_pe_rot

    # k_pe_all 需从 split 取
    compressed_kv_buf, k_pe_all_buf = latents.split([kv_lora, qk_rope], dim=-1)

    def step_cat_flash_oproj():
        q_nope, q_pe_r, k_pe_rot = step_rope()
        q_full = torch.cat([q_nope, q_pe_r], dim=-1)
        k_nope, v = kv.split([qk_nope, v_head], dim=-1)
        k_full = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, num_heads, -1)], dim=-1)
        v_fa = F.pad(v, (0, q_head - v_head))
        cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q_full.device)
        cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q_full.device)
        total_lens = cache_lens.long()
        cu_k[1:] = torch.cumsum(total_lens.to(torch.int32), dim=0)
        k_v = k_full.reshape(bs * max_len, k_full.shape[-2], k_full.shape[-1])
        v_v = v_fa.reshape(bs * max_len, v_fa.shape[-2], v_fa.shape[-1])
        attn_out = flash_attn_varlen_func(
            q_full, k_v, v_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=max_len,
            softmax_scale=gr._ds_softmax_scale, causal=False)
        attn_out = attn_out[..., :v_head].reshape(bs, num_heads * v_head)
        return F.linear(attn_out, attn._o_w, attn._o_b)

    # warmup
    for _ in range(3):
        step_store(); step_gather(); step_kvb(); step_rope(); step_cat_flash_oproj()
    torch.cuda.synchronize()

    print(f"\n=== bs=1 attention 子段 GPU 时间 ===\n", flush=True)
    t_store = bench_graph(step_store)
    print(f"  store (写新 latent):        {t_store:6.1f} us")
    t_gather = bench_graph(step_gather)
    print(f"  gather (读全部 latent):     {t_gather:6.1f} us")
    t_kvb = bench_graph(step_kvb)
    print(f"  kvb (rmsnorm+kvb_proj):     {t_kvb:6.1f} us")
    t_rope = bench_graph(step_rope)
    print(f"  rope (q_pe+k_pe):           {t_rope:6.1f} us")
    t_flash_oproj = bench_graph(step_cat_flash_oproj)
    print(f"  cat+flash+oproj:            {t_flash_oproj:6.1f} us")
    print(f"  --- 合计: {t_store+t_gather+t_kvb+t_rope+t_flash_oproj:.1f} us (整段 attention 实测 ~157us)")

    # ---- MoE 子段拆解 ----
    print(f"\n=== bs=1 MoE 子段 GPU 时间 ===\n", flush=True)
    from models.deepseek.moe import moe_forward
    mlp = moe_block.mlp
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    gr._residual[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    x_norm = gr._h_buf[:bs]
    residual = gr._residual[:bs]
    rmsnorm_residual(torch.randn(bs, H, device="cuda", dtype=engine.dtype), residual,
                     mlp._post_ln_w if hasattr(mlp,'_post_ln_w') else moe_block._post_ln_w,
                     gr._h_buf[:bs], gr._residual[:bs], 1e-6)
    # 实际用 block 的 post_ln
    rmsnorm_residual(torch.randn(bs, H, device="cuda", dtype=engine.dtype), residual,
                     moe_block._post_ln_w, gr._h_buf[:bs], gr._residual[:bs], moe_block._post_ln_eps)
    x_in = gr._h_buf[:bs]

    # gate + topk (PyTorch)
    def step_gate_topk():
        logits = F.linear(x_in, mlp._gate_w)
        return torch.topk(logits, A._top_k, dim=-1)

    # shared expert (dense SwiGLU): shared_gu [hidden, 2*s_inter], shared_d [s_inter, hidden]
    def step_shared():
        gu = x_in @ mlp._shared_gu
        gate, up = gu.chunk(2, dim=-1)
        act = F.silu(gate) * up
        return act @ mlp._shared_d

    # routed MoE (Triton grouped_gemv, 不含 shared)
    def step_routed():
        return moe_forward(x_in.reshape(-1, H), mlp._gate_w, mlp._e_gu, mlp._e_d,
                           A._top_k, A._n_experts, None, None, decode=True)

    for _ in range(3):
        step_gate_topk(); step_shared(); step_routed()
    torch.cuda.synchronize()

    t_gate = bench_graph(step_gate_topk)
    print(f"  gate+topk:                  {t_gate:6.1f} us")
    t_shared = bench_graph(step_shared)
    print(f"  shared (SwiGLU dense):      {t_shared:6.1f} us")
    t_routed = bench_graph(step_routed)
    print(f"  routed MoE (gate+topk+gemv):{t_routed:6.1f} us")
    print(f"  --- 合计: shared+routed ≈ {t_shared+t_routed:.1f} us (整段 ffn MoE 实测 ~179us, 含 norm)")


if __name__ == "__main__":
    main()
