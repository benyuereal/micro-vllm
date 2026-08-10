#!/usr/bin/env python3
"""整层 decode 各阶段 CUDA 耗时 + kernel 边界数 profile。

在一层 decode 的每个钩子（compute_qkv/attention/compute_ffn/compute_next_qkv）外
用 cuda.Event 圈住，统计各段耗时；同时用 torch.profiler 数一层内的 kernel launch 数
（kernel 边界 = execution gap 的来源）。

目的：为「单层全融合 persistent kernel」定边界——哪些段融进去、MoE data-dependent
路由怎么处理、层间 h 状态走 smem 还是 HBM。
"""
import sys
import time
import types
import torch

sys.path.insert(0, "/models/micro-vllm")

from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from torch.profiler import profile, ProfilerActivity

REGION_NAMES = ["qkv", "attention", "ffn", "next_qkv"]


def patch_layer(adapter, events):
    """圈住一层 decode 的 4 个钩子。attention 内部不再细分（见 prof_attention.py）。"""
    orig_qkv = adapter.compute_qkv
    orig_attn = adapter.attention
    orig_ffn = adapter.compute_ffn
    orig_next = adapter.compute_next_qkv

    def mark(region, fn):
        def wrapped(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            events[region].append((s, e))
            return out
        return wrapped

    adapter.compute_qkv = types.MethodType(mark("qkv", orig_qkv.__func__), adapter)
    adapter.attention = types.MethodType(mark("attention", orig_attn.__func__), adapter)
    adapter.compute_ffn = types.MethodType(mark("ffn", orig_ffn.__func__), adapter)
    adapter.compute_next_qkv = types.MethodType(mark("next_qkv", orig_next.__func__), adapter)


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    events = {r: [] for r in REGION_NAMES}
    patch_layer(engine.graph_runner.adapter, events)

    # eager
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

    print(f"Profiling decode (bs={bs}, eager) ...", flush=True)
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
    n_layers = engine.num_layers  # 27

    totals = {r: 0.0 for r in REGION_NAMES}
    counts = {r: len(events[r]) for r in REGION_NAMES}
    for r in REGION_NAMES:
        for s, e in events[r]:
            totals[r] += s.elapsed_time(e)

    layer_total = sum(totals.values())  # 一层四段（next_qkv 属于下一层入口，近似算进层内）

    print("\n" + "=" * 80)
    print(f"  bs={bs}, {decode_steps} decode steps, {n_layers} layers/step (eager)")
    print(f"  各段调用次数: {counts}  (qkv 首层1次+next_qkv 26次={n_layers})")
    print("-" * 80)
    print(f"{'region':<12}{'total(ms)':>14}{'per_step(ms)':>16}{'per_layer(us)':>16}{'%layer':>10}")
    print("-" * 80)
    for r in REGION_NAMES:
        per_step = totals[r] / decode_steps
        per_layer_us = per_step / n_layers * 1000
        pct = totals[r] / layer_total * 100 if layer_total else 0
        print(f"{r:<12}{totals[r]:14.2f}{per_step:16.2f}{per_layer_us:16.1f}{pct:10.2f}")
    print("-" * 80)
    print(f"{'LAYER':<12}{layer_total:14.2f}{layer_total/decode_steps:16.2f}{'':>16}{100.0:>10.2f}")
    print("=" * 80)
    print(f"\n一层 decode 总耗时(eager): {layer_total/decode_steps/n_layers:.3f} ms/层")
    print(f"  其中 attention: {totals['attention']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  其中 ffn(MoE):  {totals['ffn']/decode_steps/n_layers*1000:.1f} us/层")
    print(f"  其中 qkv+next:  {(totals['qkv']+totals['next_qkv'])/decode_steps/n_layers*1000:.1f} us/层")

    # 额外：用 profiler 数一层内的 kernel 数（execution gap = kernel 边界数）
    print("\n=== 一层 decode 的 kernel 边界数（profiler 统计）===")
    # 跑 5 步，统计 unique kernel 调用次数 / 5 / 27
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            batch, bt = engine.get_next_batch()
            if not batch: break
            ctx = BatchInferenceContext(len(batch), bt, batch)
            engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    ka = prof.key_averages()
    total_kernel_calls = sum(r.count for r in ka)
    # device kernel 数（排除 cpu op）
    device_calls = sum(r.count for r in ka if r.device_time_total > 0)
    print(f"  5 步 decode 总 kernel 调用: {total_kernel_calls} (含CPU op), device kernel: {device_calls}")
    print(f"  每步 kernel 调用: {total_kernel_calls/5:.0f}, 每层: {total_kernel_calls/5/n_layers:.1f}")
    print(f"  前 15 kernel（按调用数）:")
    for r in sorted(ka, key=lambda x: -x.count)[:15]:
        print(f"    {r.count:5d}× {r.key[:70]}")


if __name__ == "__main__":
    main()
