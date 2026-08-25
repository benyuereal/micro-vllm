"""W8A16 vs bf16 prefill logits 对比（同输入 token）：量化误差 vs bug。
通过 wrap sampler 捕获 prefill 的 last_logits。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "中国的首都是"


def prefill_logits(prompt):
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    eng.add_request(prompt, 1, temperature=0.0, top_p=1.0)
    captured = {}
    orig = eng.sampler
    def wrap(logits, *a, **k):
        captured["lg"] = logits.float().clone()
        return orig(logits, *a, **k)
    eng.sampler = wrap
    b, bt = eng.get_next_batch()
    ctx = BatchInferenceContext(len(b), bt, b)
    eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    eng.sampler = orig
    return captured["lg"]


def main():
    w8 = os.environ.get("MICRO_W8A16", "0") == "1"
    lg = prefill_logits(PROMPT)
    torch.save(lg, f"/tmp/w8_prefill_{'w8' if w8 else 'bf16'}.pt")
    topk = lg.topk(5)
    print(f"{'W8A16' if w8 else 'BF16'} prefill top5: " +
          " ".join(f"{t}:{v:.3f}" for t, v in zip(topk.indices.tolist(), topk.values.tolist())))


if __name__ == "__main__":
    main()
