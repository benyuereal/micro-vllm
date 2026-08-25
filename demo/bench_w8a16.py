"""W8A16 vs bf16 单用户 decode 吞吐对比（tok/s）。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 8)
OUT_TOK = 256


def main():
    mode = "W8A16" if os.environ.get("MICRO_W8A16", "0") == "1" else "BF16"
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    # warmup + graph capture
    eng.add_request(PROMPT, 16, temperature=0.0, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    # 计时 decode
    eng.add_request(PROMPT, OUT_TOK, temperature=0.0, top_p=1.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 0
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        if bt == "decode":
            n_steps += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{mode}: {n_steps} decode steps in {dt:.3f}s = {n_steps/dt:.1f} tok/s "
          f"({dt/n_steps*1000:.2f} ms/step)")


if __name__ == "__main__":
    main()
