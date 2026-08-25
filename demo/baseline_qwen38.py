"""Qwen3.8-27B W8A16 基线：engine 单用户 decode 吞吐（非 spec）。

用法：CUDA_VISIBLE_DEVICES=4 MICRO_W8A16=1 python3 demo/baseline_qwen38.py
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 8)
OUT_TOK = int(os.environ.get("OUT_TOK", "128"))


def main():
    mode = "W8A16" if os.environ.get("MICRO_W8A16", "0") == "1" else "BF16"
    print(f"mode={mode} model={MODEL} out={OUT_TOK}")
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096)
    # warmup
    eng.add_request(PROMPT, 16, temperature=0.0, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    # timed decode
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
    # 打印生成文本前 200 字符
    for seq in list(eng.scheduler.finished_sequences):
        print("gen text:", eng.tokenizer.decode(seq.output_ids, skip_special_tokens=True)[:200])
        break


if __name__ == "__main__":
    main()
