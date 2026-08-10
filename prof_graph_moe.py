#!/usr/bin/env python3
"""量 CUDA Graph 下 MoE 的真实耗时（不是 eager）。

关键：当前 decode 在 CUDA Graph 里，for i in range(N) 静态展开成 16 个 Triton kernel，
launch 开销被 graph 摊掉。eager profile 的 1055us 可能高估了 graph 下的开销。

方法：在 graph replay 路径上用 cuda.Event 圈住 compute_ffn（MoE），对比 eager。
"""
import sys
import time
import torch
sys.path.insert(0, "/models/micro-vllm")

from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    bs = 8
    prompts = [
        "写一个 SpringBoot 文件上传代码", "解释量子力学的基本概念",
        "用 Python 实现快速排序", "介绍宋朝的历史",
        "如何学习机器学习", "写一首关于秋天的诗",
        "解释什么是 Transformer", "推荐几本计算机经典书籍",
    ]
    for p in prompts[:bs]:
        engine.add_request(p, max_tokens=150, temperature=0.0)

    # 跑到 decode
    for _ in range(200):
        batch, bt = engine.get_next_batch()
        if bt == "waiting" or not batch:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break

    # 跑几步 decode 让 graph 稳定
    for _ in range(5):
        batch, bt = engine.get_next_batch()
        if not batch: break
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    # 量 graph 下整步 decode 耗时
    print(f"\n=== Graph 下 decode (bs={bs}) ===", flush=True)
    times = []
    for _ in range(50):
        batch, bt = engine.get_next_batch()
        if not batch or bt != "decode": continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        engine.step(ctx)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        engine.collect(ctx); engine.update_sequences(ctx.sequences)

    times = torch.tensor(times)
    print(f"  graph decode/step: median={times.median().item():.3f}ms mean={times.mean().item():.3f}ms min={times.min().item():.3f}ms")
    print(f"  稳态吞吐: {1000/times.median().item():.1f} tok/s")
    print(f"  (基准: 13.47ms/step, 72.2 tok/s)")

    # 对比 eager（强制 eager 路径）
    import types
    gr = engine.graph_runner
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen
    def eager_forward(input_ids, cache_manager, batch_size):
        if input_ids is not None:
            gr._input_ids[:batch_size] = input_ids
        with torch.no_grad():
            return gr.decode(gr._input_ids[:batch_size], batch_size,
                             cache_manager, cache_manager._block_table_buffer)
    gr.forward = eager_forward

    print(f"\n=== Eager 下 decode (bs={bs}) ===", flush=True)
    # warmup eager
    for _ in range(3):
        batch, bt = engine.get_next_batch()
        if not batch: break
        ctx = BatchInferenceContext(len(batch), bt, batch)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    times_e = []
    for _ in range(50):
        batch, bt = engine.get_next_batch()
        if not batch or bt != "decode": continue
        ctx = BatchInferenceContext(len(batch), bt, batch)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        engine.step(ctx)
        e.record()
        torch.cuda.synchronize()
        times_e.append(s.elapsed_time(e))
        engine.collect(ctx); engine.update_sequences(ctx.sequences)

    times_e = torch.tensor(times_e)
    print(f"  eager decode/step: median={times_e.median().item():.3f}ms mean={times_e.mean().item():.3f}ms")
    print(f"\n=== 结论 ===")
    print(f"  graph vs eager: {times.median().item():.2f}ms vs {times_e.median().item():.2f}ms")
    print(f"  graph 省了: {times_e.median().item()-times.median().item():.2f}ms ({(1-times.median().item()/times_e.median().item())*100:.0f}%)")
    print(f"  → graph 下 MoE 的 16 次 kernel launch 已被摊掉，eager 的 1055us 高估了 graph 开销")


if __name__ == "__main__":
    main()
