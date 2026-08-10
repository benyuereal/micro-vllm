#!/usr/bin/env python3
"""端到端 decode 吞吐微基准（融合 MLA + MoE）。

进程内直驱 engine 循环，CUDA event 计时稳态 decode（排除 prefill + 首轮热启动）。
decode 路径已默认走融合 kernel（kernel/mla.py + kernel/moe.py + kernel/pre_mla.py），无开关。
用法：
  python3 bench_tl_mla_e2e.py
"""
import sys, time, torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = "/models/DeepSeek-V2-Lite"
PROMPT = "请详细解释 Transformer 架构中多头自注意力机制的完整计算流程，包括 Q K V 矩阵的生成、缩放点积注意力、softmax 归一化、多头拼接和输出投影。"
WARMUP_GEN = 60      # 排除首轮热启动
MEASURE_GEN = 200    # 计量 token 数


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine(MODEL, max_batch_size=40)
    engine.add_request(PROMPT, max_tokens=WARMUP_GEN + MEASURE_GEN + 10, temperature=0.0)

    # prefill
    while True:
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break

    # warmup decode
    for _ in range(WARMUP_GEN):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    torch.cuda.synchronize()

    # measure decode（CUDA event 包住 N 步）
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_GEN)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_GEN)]
    n = 0
    for i in range(MEASURE_GEN):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        starts[i].record()
        engine.step(ctx)
        ends[i].record()
        engine.collect(ctx); engine.update_sequences(ctx.sequences)
        n += 1
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts[:n], ends[:n])]
    times.sort()
    total_ms = sum(times)
    import statistics
    med = statistics.median(times)
    mean = statistics.mean(times)
    tps = n / (total_ms / 1000.0)
    steady_tps = 1000.0 / med
    print(f"\n=== fused MLA + MoE ===")
    print(f"  measured tokens : {n}")
    print(f"  total wall      : {total_ms:.1f} ms")
    print(f"  mean step       : {mean:.3f} ms")
    print(f"  median step     : {med:.3f} ms")
    print(f"  throughput      : {tps:.1f} tok/s")
    print(f"  steady(1/median): {steady_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
