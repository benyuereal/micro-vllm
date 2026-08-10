#!/usr/bin/env python3
"""MoE decode 内部细分 profile：gate/softmax_topk/grouped_gemv_loop/shared/norm。

把 moe_forward 的 decode 路径每段用 cuda.Event 圈住，量化 1161us/层 的 ffn 里 MoE
routing 各段占比，判断 data-dependent 路由在 persistent kernel 里的可行性。
"""
import sys
import time
import types
import torch

sys.path.insert(0, "/models/micro-vllm")

from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
import models.deepseek.moe as moe_mod

REGION_NAMES = ["gate", "topk", "gemv_loop", "shared", "norm"]


def patched_moe(events):
    """返回一个替换 moe_forward 的函数，内部圈住各段。"""
    import torch.nn.functional as F
    from kernel.grouped_gemv import grouped_gate_up, grouped_down

    def moe_forward(x, gate_weight, e_gu, e_d, top_k, n_experts,
                    shared_gu=None, shared_d=None, decode=False):
        N = x.shape[0]
        hidden = x.shape[1]

        def rec():
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev

        s_gate = rec()
        logits = F.linear(x, gate_weight)
        scores = logits.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        e_gate = rec(); events["gate"].append((s_gate, e_gate))

        s_topk = rec()
        topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
        e_topk = rec(); events["topk"].append((s_topk, e_topk))

        flat_idx = topk_idx.reshape(-1)
        flat_w = topk_weight.reshape(-1)

        if decode:
            out = torch.empty(N, hidden, dtype=x.dtype, device=x.device)
            w_ones = torch.ones(top_k, dtype=x.dtype, device=x.device)
            s_loop = rec()
            for i in range(N):
                idx_i = flat_idx[i * top_k:(i + 1) * top_k].to(torch.int64)
                w_i = flat_w[i * top_k:(i + 1) * top_k]
                gu = grouped_gate_up(x[i:i + 1], e_gu, idx_i)
                gate, up = gu.chunk(2, dim=-1)
                act = F.silu(gate) * up * w_i.unsqueeze(-1).to(gu.dtype)
                out[i:i + 1] = grouped_down(act, e_d, idx_i, w_ones)
            e_loop = rec(); events["gemv_loop"].append((s_loop, e_loop))
        else:
            # prefill: 不圈 timing，用原逻辑
            x_rep = x.unsqueeze(1).expand(N, top_k, hidden).reshape(N * top_k, hidden)
            order = flat_idx.argsort()
            sorted_idx = flat_idx[order]
            sorted_x = x_rep[order]
            sorted_w = flat_w[order]
            counts = torch.bincount(sorted_idx, minlength=n_experts)
            out_rep = torch.empty_like(sorted_x)
            cum = 0
            counts_list = counts.tolist()
            for ei, cnt in enumerate(counts_list):
                if cnt == 0:
                    continue
                seg = sorted_x[cum:cum + cnt]
                gu = e_gu[ei]
                d = e_d[ei]
                gate_up = seg @ gu.t()
                gate, up = gate_up.chunk(2, dim=-1)
                act = F.silu(gate) * up
                out_rep[cum:cum + cnt] = act @ d.t()
                cum += cnt
            inv_order = order.argsort()
            out_rep = out_rep[inv_order]
            out = (out_rep.view(N, top_k, hidden) *
                   sorted_w[inv_order].view(N, top_k, 1).to(out_rep.dtype)).sum(dim=1)

        if shared_gu is not None:
            s_sh = rec()
            gate_up = x @ shared_gu
            gate, up = gate_up.chunk(2, dim=-1)
            out = out + (F.silu(gate) * up) @ shared_d
            e_sh = rec(); events["shared"].append((s_sh, e_sh))

        return out
    return moe_forward


def patch_norm(adapter, events):
    """圈住 compute_ffn/compute_qkv/compute_next_qkv 里的 rmsnorm。
    直接 wrap adapter 的 rmsnorm 调用太分散；改为圈 compute_ffn 内部前后的 norm。
    简化：单独量 moe_forward 前的 input_norm（在 compute_ffn 里）。"""
    # compute_ffn 结构: input_norm(h) -> moe -> return. 我们圈整个 compute_ffn 减去 moe 即 norm。
    orig_ffn = adapter.compute_ffn
    def wrapped(self, *a, **kw):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_ffn(*a, **kw)
        e.record()
        events["norm"].append((s, e))  # 这里是整个 ffn，moe 段会从中减
        return out
    adapter.compute_ffn = types.MethodType(wrapped, adapter)


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    events = {r: [] for r in REGION_NAMES}
    # 替换 moe 模块里的 moe_forward，使 adapter 调到带 timing 的版本
    moe_mod.moe_forward = patched_moe(events)
    # adapter 里是 from ... import moe_forward 还是 import？检查
    import models.deepseek.adapter as ad
    # adapter 用的是 moe_forward 名字引用，需替换 adapter 模块命名空间
    if hasattr(ad, "moe_forward"):
        ad.moe_forward = moe_mod.moe_forward

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
        "写一个 SpringBoot 文件上传代码", "解释量子力学的基本概念",
        "用 Python 实现快速排序", "介绍宋朝的历史",
        "如何学习机器学习", "写一首关于秋天的诗",
        "解释什么是 Transformer", "推荐几本计算机经典书籍",
    ]
    for p in prompts[:bs]:
        engine.add_request(p, max_tokens=150, temperature=0.0)

    for _ in range(200):
        batch, bt = engine.get_next_batch()
        if bt == "waiting" or not batch:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break

    for r in REGION_NAMES:
        events[r].clear()

    print(f"Profiling MoE decode (bs={bs}, eager) ...", flush=True)
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
        if decode_steps >= N_STEPS: break

    torch.cuda.synchronize()
    n_layers = engine.num_layers

    totals = {r: 0.0 for r in REGION_NAMES}
    counts = {r: len(events[r]) for r in REGION_NAMES}
    for r in REGION_NAMES:
        for s, e in events[r]:
            totals[r] += s.elapsed_time(e)

    # gate+topk 是每层 1 次(N=8 一次算)；gemv_loop/shared 每层 1 次
    # counts 应该 = decode_steps * n_layers
    print("\n" + "=" * 80)
    print(f"  bs={bs}, {decode_steps} decode steps, {n_layers} layers/step (eager)")
    print(f"  调用次数: {counts}")
    print("-" * 80)
    print(f"{'region':<14}{'total(ms)':>12}{'per_step(ms)':>14}{'per_layer(us)':>16}{'calls':>8}")
    print("-" * 80)
    for r in ["gate", "topk", "gemv_loop", "shared"]:
        per_step = totals[r] / decode_steps
        per_layer_us = per_step / n_layers * 1000
        print(f"{r:<14}{totals[r]:12.2f}{per_step:14.2f}{per_layer_us:16.1f}{counts[r]:8d}")
    moe_total = sum(totals[r] for r in ["gate", "topk", "gemv_loop", "shared"])
    print("-" * 80)
    print(f"{'MoE_total':<14}{moe_total:12.2f}{moe_total/decode_steps:14.2f}{moe_total/decode_steps/n_layers*1000:16.1f}{counts['gate']:8d}")
    print("=" * 80)
    print(f"\nMoE decode 每层: {moe_total/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  gate(linear+softmax): {totals['gate']/decode_steps/n_layers*1000:.1f} us ({totals['gate']/moe_total*100:.1f}%)")
    print(f"  topk:                 {totals['topk']/decode_steps/n_layers*1000:.1f} us ({totals['topk']/moe_total*100:.1f}%)")
    print(f"  gemv_loop(N×2 kernel):{totals['gemv_loop']/decode_steps/n_layers*1000:.1f} us ({totals['gemv_loop']/moe_total*100:.1f}%)")
    print(f"  shared(2 matmul+silu):{totals['shared']/decode_steps/n_layers*1000:.1f} us ({totals['shared']/moe_total*100:.1f}%)")
    print(f"\n  gemv_loop 占 MoE {totals['gemv_loop']/moe_total*100:.1f}% — 这是 N={bs} 个 token 各调 2 个 triton kernel 的开销")


if __name__ == "__main__":
    main()
