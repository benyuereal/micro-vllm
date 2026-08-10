#!/usr/bin/env python3
"""attention() 内部各阶段 CUDA 耗时 profiling（CUDA event 累计计时）。

eager 路径下，每个 region 每次调用用一对独立 cuda.Event 圈住，所有 event 存进列表，
全部跑完后统一 synchronize 再读 elapsed_time。避免 per-step 重绑闭包与 event 复用问题。
op 间相对占比在 graph/eager 下基本一致（graph 只省 launch 开销）。
"""
import sys
import time
import types
import torch

sys.path.insert(0, "/models/micro-vllm")

from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
import kernel.rmsnorm as rmsmod
from flash_attn import flash_attn_varlen_func

REGION_NAMES = ["store", "gather", "kvb", "rope", "flash", "oproj"]


def patch_attention(adapter, events):
    """events: dict region -> list of (start_evt, end_evt)，每次调用 append 一对。"""
    rmsnorm = rmsmod.rmsnorm

    def patched(self, x_normed, block, layer_idx, bs, graph, cache_manager, block_table):
        attn = block.self_attn
        q = attn._q_cache
        compressed_kv_new = attn._compressed_kv
        k_pe_new = attn._k_pe
        k_cache, v_cache = cache_manager.get(layer_idx)
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]
        new_pos = (cache_lens - 1).long().clamp(min=0)

        def mark(r):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            events[r].append((s, e))
            return e  # 调用方在 region 结束时 e.record()

        # (1) store
        e = mark("store")
        latent_new = torch.cat([compressed_kv_new, k_pe_new], dim=-1).view(bs, 1, 1, self._latent_dim)
        slots = self._decode_slots(block_table, new_pos, bs, cache_manager.block_size)
        self._store_latent_batch(latent_new, k_cache, v_cache, slots, cache_manager.block_size)
        e.record()

        # (2) gather
        e = mark("gather")
        total_lens = cache_lens.long()
        max_len = graph._cur_bucket_maxlen
        block_size = cache_manager.block_size
        bt = block_table[:bs].long()
        t_idx = torch.arange(max_len, device=bt.device)
        blk_id = bt[:, t_idx // block_size].clamp(min=0)
        n_slots = k_cache.shape[0] * block_size
        slots = (blk_id * block_size + (t_idx % block_size)).clamp(min=0, max=n_slots - 1)
        k_flat = k_cache.reshape(-1, self._latent_dim)
        latents = k_flat[slots.reshape(-1)].view(bs, max_len, self._latent_dim)
        e.record()

        # (3) kvb
        e = mark("kvb")
        compressed_kv, k_pe_all = latents.split([self._kv_lora_rank, self._qk_rope], dim=-1)
        ckv = rmsnorm(compressed_kv.reshape(-1, self._kv_lora_rank), attn._kva_ln_w, attn._kva_ln_eps)
        kv = torch.nn.functional.linear(ckv, attn._kvb_w).view(
            bs, max_len, self._num_heads, self._qk_nope + self._v_head)
        k_nope, v = kv.split([self._qk_nope, self._v_head], dim=-1)
        e.record()

        # (4) rope
        e = mark("rope")
        cos, sin = self._rope_pool(graph, k_cache.device)
        q_nope, q_pe = q.split([self._qk_nope, self._qk_rope], dim=-1)
        cos_q = cos[new_pos].unsqueeze(1); sin_q = sin[new_pos].unsqueeze(1)
        q_pe = self._apply_rope(q_pe, cos_q, sin_q)
        k_pos = torch.arange(max_len, device=k_pe_all.device).unsqueeze(0)
        k_pe_rot = self._apply_rope(k_pe_all, cos[k_pos], sin[k_pos])
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, self._num_heads, -1)], dim=-1)
        v_fa = torch.nn.functional.pad(v, (0, self._q_head - self._v_head))
        e.record()

        # (5) flash
        e = mark("flash")
        cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q_full.device)
        cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q_full.device)
        cu_k[1:] = torch.cumsum(total_lens.to(torch.int32), dim=0)
        k_v = k_full.reshape(bs * max_len, *k_full.shape[-2:])
        v_v = v_fa.reshape(bs * max_len, *v_fa.shape[-2:])
        attn_out = flash_attn_varlen_func(
            q_full, k_v, v_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=max_len,
            softmax_scale=graph._ds_softmax_scale, causal=False)
        attn_out = attn_out[..., :self._v_head].reshape(bs, self._num_heads * self._v_head)
        e.record()

        # (6) oproj
        e = mark("oproj")
        out = torch.nn.functional.linear(attn_out, attn._o_w, attn._o_b)
        e.record()
        return out

    adapter.attention = types.MethodType(patched, adapter)


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    events = {r: [] for r in REGION_NAMES}
    # decode 走 graph_runner.adapter（与 engine.adapter 是不同对象），patch 这个
    patch_attention(engine.graph_runner.adapter, events)

    # eager：forward → decode 直接跑
    gr = engine.graph_runner
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen
    def eager_forward(input_ids, cache_manager, batch_size):
        if input_ids is not None:
            gr._input_ids[:batch_size] = input_ids
        with torch.no_grad():
            return gr.decode(gr._input_ids[:batch_size], batch_size,
                             cache_manager, cache_manager._block_table_buffer)
    gr.forward = eager_forward

    bs = 8
    prompts = [
        "写一个 SpringBoot 文件上传代码",
        "解释量子力学的基本概念",
        "用 Python 实现快速排序",
        "介绍宋朝的历史",
        "如何学习机器学习",
        "写一首关于秋天的诗",
        "解释什么是 Transformer",
        "推荐几本计算机经典书籍",
    ]
    for p in prompts[:bs]:
        engine.add_request(p, max_tokens=150, temperature=0.0)

    # 跑到 decode（清空 events 里 prefill 阶段误记的）
    for _ in range(200):
        batch, bt = engine.get_next_batch()
        if bt == "waiting" or not batch:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill":
            break

    # 清空 prefill 阶段记录（prefill 走 prefill_layer 不调 attention，应已为空，保险起见清）
    for r in REGION_NAMES:
        events[r].clear()

    print(f"Profiling decode (bs={bs}, eager, max_len=1024) ...", flush=True)
    N_STEPS = 40
    decode_steps = 0
    for step in range(N_STEPS + 50):
        batch, bt = engine.get_next_batch()
        if not batch: break
        if bt != "decode":
            ctx = BatchInferenceContext(len(batch), bt, batch)
            engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
            continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        decode_steps += 1
        if decode_steps >= N_STEPS:
            break

    torch.cuda.synchronize()

    n_layers = engine.num_layers  # 27
    totals = {r: 0.0 for r in REGION_NAMES}
    for r in REGION_NAMES:
        for s, e in events[r]:
            totals[r] += s.elapsed_time(e)  # ms

    calls = {r: len(events[r]) for r in REGION_NAMES}
    attn_total = sum(totals.values())

    print("\n" + "=" * 80)
    print(f"  bs={bs}, {decode_steps} decode steps, {n_layers} layers/step, max_len=1024 (eager)")
    print(f"  region 调用次数: {calls}")
    print("-" * 80)
    print(f"{'region':<12}{'total(ms)':>14}{'per_call(us)':>16}{'%attn':>10}")
    print("-" * 80)
    for r in REGION_NAMES:
        per_call_us = (totals[r] / calls[r] * 1000) if calls[r] else 0
        pct = totals[r] / attn_total * 100 if attn_total else 0
        print(f"{r:<12}{totals[r]:14.2f}{per_call_us:16.1f}{pct:10.2f}")
    print("-" * 80)
    per_step_attn = attn_total / decode_steps if decode_steps else 0
    print(f"{'ATTENTION':<12}{attn_total:14.2f}{'':>16}{100.0:>10.2f}")
    print("=" * 80)
    print(f"\nattention 每步总耗时(eager): {per_step_attn:.2f} ms (含 {n_layers} 层)")
    print(f"attention 每层每步: {per_step_attn/n_layers:.3f} ms")
    print(f"  - store:  {totals['store']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  - gather: {totals['gather']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  - kvb:    {totals['kvb']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  - rope:   {totals['rope']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  - flash:  {totals['flash']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  - oproj:  {totals['oproj']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"\n注: eager 计时含 launch 开销；graph 下 launch 被省，但各 op 相对占比基本不变。")


if __name__ == "__main__":
    main()
