"""HF 逐步 forward，打印每个 decode 位置的 top-5 logits + top1-top2 gap，
判断 token 6 分叉是 bf16 平局还是真 bug。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    model.eval()
    ids = tok.encode(PROMPT, add_special_tokens=True)
    cur = torch.tensor([ids], device="cuda:0")
    print(f"prompt: {ids}")
    for step in range(12):
        with torch.no_grad():
            out = model(cur)
        logits = out.logits[0, -1].float()
        topk = logits.topk(5)
        gap = (topk.values[0] - topk.values[1]).item()
        print(f"step {step} (pos {len(ids)+step}): top1={topk.indices[0].item()} "
              f"v={topk.values[0].item():.3f} gap={gap:.4f} "
              f"top5={list(zip(topk.indices.tolist(), [f'{v:.2f}' for v in topk.values.tolist()]))}")
        cur = torch.cat([cur, topk.indices[0:1].unsqueeze(0)], dim=1)


if __name__ == "__main__":
    main()
