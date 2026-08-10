#!/usr/bin/env python3
"""精准量 graph 下 MoE 的真实 GPU 时间：单独 capture 一个只跑 MoE 的 graph。

这样测的是 CUDA Graph 下 27 层 MoE 的纯 GPU 时间，排除 attention/norm/sampler。
对比 eager MoE (1055us/层 × 27 = 28.5ms) 看 graph 摊掉了多少。
"""
import sys
import time
import torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from models.deepseek.moe import moe_forward


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    bs = 8
    prompts = [
        "写SpringBoot文件上传代码", "解释量子力学", "Python快速排序", "宋朝历史",
        "学机器学习", "秋天诗", "Transformer", "计算机书",
    ]
    for p in prompts[:bs]:
        engine.add_request(p, max_tokens=150, temperature=0.0)
    for _ in range(200):
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b: time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break
    for _ in range(5):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    # 拿到第一个 MoE 层的权重（DeepSeek-V2-Lite 前 3 层 dense，第 4 层起 MoE）
    adapter = engine.graph_runner.adapter
    blocks = adapter.blocks(engine.graph_runner.model)
    mlp = None
    for block in blocks:
        if getattr(block.mlp, "_is_moe", False):
            mlp = block.mlp
            break
    assert mlp is not None, "no MoE layer found"
    H = adapter._hidden

    # 准备输入 x [bs, H]（随机，只测 MoE kernel 时间）
    x = torch.randn(bs, H, device="cuda", dtype=engine.dtype)

    # warmup MoE
    for _ in range(5):
        moe_forward(x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                    adapter._top_k, adapter._n_experts,
                    mlp._shared_gu, mlp._shared_d, decode=True)

    # === 1. Eager 单层 MoE ===
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    N_ITER = 100
    s.record()
    for _ in range(N_ITER):
        moe_forward(x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                    adapter._top_k, adapter._n_experts,
                    mlp._shared_gu, mlp._shared_d, decode=True)
    e.record(); torch.cuda.synchronize()
    eager_us = s.elapsed_time(e) / N_ITER * 1000
    print(f"\nEager 单层 MoE (bs={bs}): {eager_us:.1f} us")

    # === 2. Graph 单层 MoE ===
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = moe_forward(x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                          adapter._top_k, adapter._n_experts,
                          mlp._shared_gu, mlp._shared_d, decode=True)
    # warmup graph
    for _ in range(10): g.replay()
    s.record()
    for _ in range(N_ITER): g.replay()
    e.record(); torch.cuda.synchronize()
    graph_us = s.elapsed_time(e) / N_ITER * 1000
    print(f"Graph 单层 MoE (bs={bs}): {graph_us:.1f} us")
    print(f"  ×27 层 = {graph_us*27/1000:.2f} ms")
    print(f"  graph vs eager: {graph_us/eager_us*100:.0f}% (graph 省了 {(1-graph_us/eager_us)*100:.0f}%)")

    # === 3. 24 层 graph MoE（真实场景：DeepSeek-V2-Lite 27 层中 24 层是 MoE）===
    N_MOE_LAYERS = 24
    g27 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g27):
        for li in range(N_MOE_LAYERS):
            moe_forward(x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                        adapter._top_k, adapter._n_experts,
                        mlp._shared_gu, mlp._shared_d, decode=True)
    for _ in range(10): g27.replay()
    s.record()
    for _ in range(N_ITER): g27.replay()
    e.record(); torch.cuda.synchronize()
    g27_us = s.elapsed_time(e) / N_ITER * 1000
    print(f"\nGraph {N_MOE_LAYERS} 层 MoE: {g27_us:.1f} us = {g27_us/1000:.2f} ms")
    print(f"  每层: {g27_us/N_MOE_LAYERS:.1f} us")
    print(f"  基准 13.47ms/step, MoE 占 {g27_us/1000/13.47*100:.1f}%")
    print(f"\n=== 结论 ===")
    print(f"  graph 下 MoE 每层 {g27_us/N_MOE_LAYERS:.0f}us, {N_MOE_LAYERS}层 {g27_us/1000:.1f}ms")
    print(f"  eager 下 MoE 每层 ~1055us, graph 下 {g27_us/N_MOE_LAYERS:.0f}us → graph 摊掉 {(1-g27_us/N_MOE_LAYERS/1055)*100:.0f}%")
    print(f"  → TileLang 要 beat 的目标是 graph 下的 {g27_us/N_MOE_LAYERS:.0f}us/层, 不是 eager 的 1055us")


if __name__ == "__main__":
    main()
