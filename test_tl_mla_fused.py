#!/usr/bin/env python3
"""standalone 正确性验证：fused MLA kernel vs 当前 attention 的 flash 输出。

构造真实 latent cache + q，跑 fused kernel，对比当前 adapter.attention() 的 attn_out（oproj 前）。
"""
import sys, time, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.adapter import rmsnorm
from kernel.tilelang_mla import _get_kernel


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

    # ---- 跑当前 attention，拿 attn_out（oproj 前）作参考 ----
    q = attn._q_cache  # [bs, H, q_head=192]
    k_cache, v_cache = cm.get(layer_idx)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    new_pos = (cache_lens - 1).long().clamp(min=0)
    max_len = gr._cur_bucket_maxlen  # 1024
    block_size = cm.block_size  # 256
    kv_lora = A._kv_lora_rank; qk_rope = A._qk_rope
    qk_nope = A._qk_nope; v_head = A._v_head; num_heads = A._num_heads
    q_head = A._q_head; latent_dim = A._latent_dim

    # 先 store 新 latent（和当前 attention 步骤1一致）
    compressed_kv_new = attn._compressed_kv
    k_pe_new = attn._k_pe
    latent_new = torch.cat([compressed_kv_new, k_pe_new], dim=-1).view(bs, 1, 1, latent_dim)
    slots = A._decode_slots(bt_buf, new_pos, bs, block_size)
    A._store_latent_batch(latent_new, k_cache, v_cache, slots, block_size)

    # 当前 attention 的 gather+kvb+rope+cat+flash（复刻 adapter，拿 oproj 前的 attn_out）
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
    attn_out_ref = attn_out_ref[..., :v_head]  # [bs, H, v_head]
    print(f"ref attn_out: shape={tuple(attn_out_ref.shape)}")

    # ---- 构造 fused kernel 输入 ----
    # weight-absorption: A[h] = Q_nope[h] @ kvb_w_kn[h]（per-head 吸收 k_nope 权重）
    # kvb_w [H*256, 512] → view [H, 256, 512]; kn = [:, :128, :], v = [:, 128:256, :]
    kvb_w_full = attn._kvb_w.view(num_heads, qk_nope + v_head, kv_lora)  # [H, 256, 512]
    kvb_w_kn = kvb_w_full[:, :qk_nope, :].contiguous()      # [H, 128, 512]
    kvb_w_v = kvb_w_full[:, qk_nope:, :].contiguous()       # [H, 128, 512]
    # A[bs, H, kv_lora] = einsum('bhd,hdk->bhk', Q_nope, kvb_w_kn)
    A_in = torch.einsum('bhd,hdk->bhk', q_nope.float(), kvb_w_kn.float()).to(engine.dtype).contiguous()
    Q_pe_in = q_pe_r.contiguous()
    Latent_flat = k_cache.reshape(-1, 1, latent_dim).contiguous()
    bt_kernel = bt_buf[:bs].to(torch.int32).contiguous()
    cache_seqlens_in = cache_lens.to(torch.int32).contiguous()
    cos_full, sin_full = A._rope_pool(gr, k_cache.device)
    cos_k_in = cos_full[:max_len].contiguous()
    sin_k_in = sin_full[:max_len].contiguous()

    softmax_scale = gr._ds_softmax_scale
    block_N = 64
    num_split = 4
    n_slots = k_cache.shape[0] * block_size   # paged cache 总槽位
    try:
        kernel = _get_kernel(bs, num_heads, max_len, kv_lora, qk_rope, qk_nope, v_head,
                             block_size, softmax_scale, engine.dtype, n_slots,
                             block_N=block_N, num_split=num_split)
    except Exception as ex:
        print(f"kernel 编译失败: {ex}")
        import traceback; traceback.print_exc()
        return

    try:
        out = kernel(A_in, Q_pe_in, Latent_flat, bt_kernel, cache_seqlens_in,
                     attn._kva_ln_w, kvb_w_v, cos_k_in, sin_k_in)
        print(f"fused out: shape={tuple(out.shape)}")
        diff = (out.float() - attn_out_ref.float()).abs()
        print(f"max_diff={diff.max().item():.4f}  mean_diff={diff.mean().item():.4f}")
        rel = diff.max().item() / attn_out_ref.float().abs().max().item()
        print(f"rel_max={rel:.4f}  OK={rel < 0.05}")
    except Exception as ex:
        print(f"kernel 运行失败: {ex}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
