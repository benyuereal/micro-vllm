"""逐步对比 micro vs HF 的 logits（中文 prompt），看分叉位置 logit 差多大。"""
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

    # 逐步跑 micro，抓每步 logits（单 seq，row 0）
    micro_step_logits = []
    micro_next = []
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        if bt == "decode":
            # ctx.logits [bs, vocab]，单 seq row 0
            micro_step_logits.append(ctx.logits[0].float().clone())
            micro_next.append(ctx.logits[0].argmax().item())

    # HF 逐步（用 micro 生成的 token 作为前缀，保证同前缀对比）
    # 先拿 micro 完整序列
    micro_ids = list(ids)
    for seq in list(eng.scheduler.finished_sequences) + list(eng.scheduler.running_sequences):
        if list(seq.input_ids) == ids:
            micro_ids = list(seq.input_ids) + list(seq.output_ids)
            break
    print("micro ids:", micro_ids)

    # micro prefill 已采样 token0（micro_ids[len(ids)]）。decode step i 的前缀 =
    # ids + micro_ids[len(ids):len(ids)+i+1]。HF 须从「含 prefill token」的前缀起步。
    cur = torch.tensor([ids + [micro_ids[len(ids)]]], device="cuda:0")
    print("pos  hf_top1  micro_top1  logit_maxdiff  top1_logit_diff  hf_gap")
    for step in range(len(micro_step_logits)):
        with torch.no_grad():
            out = hf(cur)
        hlogits = out.logits[0, -1].float()
        mlogits = micro_step_logits[step]
        maxdiff = (hlogits - mlogits).abs().max().item()
        htop = hlogits.argmax().item()
        mtop = micro_next[step]
        htopv = hlogits[htop].item()
        mtopv = mlogits[htop].item()
        topk = hlogits.topk(2)
        gap = (topk.values[0] - topk.values[1]).item()
        mark = " " if htop == mtop else "X"
        print(f"{len(ids)+1+step:3d} {mark} {htop:7d} {mtop:11d}  {maxdiff:12.4f}  {htopv-mtopv:14.4f}  {gap:8.4f}")
        cur = torch.cat([cur, torch.tensor([[mtop]], device="cuda:0")], dim=1)


if __name__ == "__main__":
    main()
