"""W8A16 vs bf16(micro) logit 对比：确认 W8A16 分叉是量化误差而非 bug。
bf16 micro 已对齐 HF，故 W8A16 与 bf16 的 logit 差 = 量化误差。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "中国的首都是"


def run_micro(prompt, n):
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    eng.add_request(prompt, n, temperature=0.0, top_p=1.0)
    step_logits = []
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        if bt == "decode":
            step_logits.append(ctx.logits[0].float().clone())
    return step_logits


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    ids = tok.encode(PROMPT, add_special_tokens=True)
    w8 = os.environ.get("MICRO_W8A16", "0") == "1"
    print(f"MICRO_W8A16={w8}")
    sl = run_micro(PROMPT, 6)
    # 打印每步 top-3 + 与「另一模式」对比由外部做；这里只打印本模式 top-3
    for i, lg in enumerate(sl):
        topk = lg.topk(3)
        print(f"  step{i} pos{len(ids)+1+i}: " +
              " ".join(f"{t}:{v:.3f}" for t, v in zip(topk.indices.tolist(), topk.values.tolist())))


if __name__ == "__main__":
    main()
