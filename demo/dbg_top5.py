"""分叉位置（pos 4）打印 micro vs HF 的 top-5 logits，看差异结构。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "中国的首都是"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    hf.eval()
    ids = tok.encode(PROMPT, add_special_tokens=True)

    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    eng.add_request(PROMPT, 8, temperature=0.0, top_p=1.0)
    micro_step_logits = []
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        if bt == "decode":
            micro_step_logits.append(ctx.logits[0].float().clone())

    # 前缀 [ids, 98116, 3709] → pos 4 的 logits
    prefix = ids + [98116, 3709]
    cur = torch.tensor([prefix], device="cuda:0")
    with torch.no_grad():
        out = hf(cur)
    hlogits = out.logits[0, -1].float()
    mlogits = micro_step_logits[1]  # decode step 1 = pos 4

    print("=== HF top-8 @ pos4 ===")
    hk = hlogits.topk(8)
    for i in range(8):
        t = hk.indices[i].item()
        print(f"  hf {t:7d} {hk.values[i].item():8.3f}   micro={mlogits[t].item():8.3f}  diff={mlogits[t].item()-hk.values[i].item():8.3f}")
    print("=== micro top-8 @ pos4 ===")
    mk = mlogits.topk(8)
    for i in range(8):
        t = mk.indices[i].item()
        print(f"  micro {t:7d} {mk.values[i].item():8.3f}   hf={hlogits[t].item():8.3f}  diff={mk.values[i].item()-hlogits[t].item():8.3f}")
    # 整体统计
    d = (mlogits - hlogits)
    print(f"\nlogit diff: mean={d.mean().item():.4f} std={d.std().item():.4f} max={d.abs().max().item():.4f}")
    print(f"  hf mean={hlogits.mean().item():.4f} micro mean={mlogits.mean().item():.4f}")
    print(f"  hf std={hlogits.std().item():.4f} micro std={mlogits.std().item():.4f}")


if __name__ == "__main__":
    main()
