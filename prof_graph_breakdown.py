#!/usr/bin/env python3
"""用 torch.profiler 测 graph replay 下各 kernel 类别的真实 GPU 时间。

在 graph 模式下 step，profiler 会记录 replay 时的 kernel。
按 kernel 名前缀分类：grouped_gate_up/grouped_down (MoE), flash_attn, elementwise, gemm 等。
"""
import sys
import time
import torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from torch.profiler import profile, ProfilerActivity, record_function


def classify(name):
    n = name.lower()
    if "grouped_gate_up" in n or "grouped_down" in n: return "MoE_grouped_gemv"
    if "silu" in n or "sigmoid" in n: return "silu"
    if "flash" in n or "flashattn" in n: return "flash_attn"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n: return "elementwise"
    if "gemm" in n or "gemv" in n or "mma" in n: return "gemm"
    if "copy" in n or "memcpy" in n: return "memcpy"
    if "softmax" in n: return "softmax"
    if "topk" in n: return "topk"
    if "reduce" in n or "sum" in n: return "reduce"
    if "scatter" in n or "gather" in n or "index" in n: return "gather_scatter"
    if "rmsnorm" in n or "rms" in n: return "rmsnorm"
    if "empty" in n or "fill" in n or "zero" in n: return "fill"
    return "other"


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    bs = 8
    prompts = [
        "写一个SpringBoot文件上传代码", "解释量子力学基本概念",
        "用Python实现快速排序", "介绍宋朝历史",
        "如何学习机器学习", "写一首关于秋天的诗",
        "解释什么是Transformer", "推荐几本计算机经典书籍",
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

    # warmup decode
    for _ in range(10):
        batch, bt = engine.get_next_batch()
        if not batch: break
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    # profile 10 步 graph decode
    print("profiling 10 graph decode steps ...", flush=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            batch, bt = engine.get_next_batch()
            if not batch: break
            ctx = BatchInferenceContext(len(batch), bt, batch)
            engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    ka = prof.key_averages()
    # 只看 device kernel
    cats = {}
    total = 0
    for r in ka:
        if r.device_time_total > 0:
            c = classify(r.key)
            cats.setdefault(c, [0, 0])
            cats[c][0] += r.device_time_total / 1000  # us
            cats[c][1] += r.count
            total += r.device_time_total / 1000

    print(f"\n=== Graph replay kernel 分解 (10 步 decode) ===")
    print(f"{'category':<22}{'gpu_us':>12}{'%':>8}{'count':>10}")
    print("-" * 52)
    for c, (us, cnt) in sorted(cats.items(), key=lambda x: -x[1][0]):
        print(f"{c:<22}{us:12.0f}{us/total*100:8.1f}{cnt:10d}")
    print("-" * 52)
    print(f"{'TOTAL':<22}{total:12.0f}{100.0:8.1f}")
    print(f"\n每步 decode GPU 时间: {total/10:.0f} us = {total/10/1000:.2f} ms")
    print(f"每步 kernel 数: {sum(c[1] for c in cats.values())//10}")
    print(f"\nMoE_grouped_gemv 占 {cats.get('MoE_grouped_gemv',[0,0])[0]/total*100:.1f}%")
    print(f"flash_attn 占 {cats.get('flash_attn',[0,0])[0]/total*100:.1f}%")


if __name__ == "__main__":
    main()
