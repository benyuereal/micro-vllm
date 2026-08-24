"""Qwen3.5-0.8B greedy 对齐测试：micro-vllm vs HF transformers。

用法：
  CUDA_VISIBLE_DEVICES=4 python3 demo/align_qwen35.py

跑两条 prompt（英文 + 中文），greedy（temp=0），逐 token 对比 HF。
HF 用 model.generate(do_sample=False) 作为唯一正确性基准。
"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
N_TOKENS = int(os.environ.get("N_TOKENS", "24"))

PROMPTS = [
    "The capital of France is",
    "中国的首都是",
]


def run_hf(prompt, n):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    model.eval()
    ids = tok.encode(prompt, add_special_tokens=True)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device="cuda:0"),
                             do_sample=False, max_new_tokens=n,
                             pad_token_id=tok.eos_token_id)
    gen = out[0].tolist()[len(ids):]
    return ids, gen, tok


def run_micro(prompt, n):
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    eng.add_request(prompt, n, temperature=0.0, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    # 找刚跑的 seq（finished 或 running 里 input_ids 匹配 prompt 的）
    tok = eng.tokenizer
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    for seq in list(eng.scheduler.finished_sequences) + list(eng.scheduler.running_sequences):
        if list(seq.input_ids) == prompt_ids:
            return prompt_ids, list(seq.output_ids), tok
    raise RuntimeError("micro: 找不到对应 seq")


def main():
    ok_all = True
    for prompt in PROMPTS:
        print(f"\n===== prompt: {prompt!r} =====")
        hf_ids, hf_gen, tok = run_hf(prompt, N_TOKENS)
        print(f"HF prompt_ids: {hf_ids}")
        print(f"HF gen ({len(hf_gen)}): {hf_gen}")
        print(f"HF text: {tok.decode(hf_gen)!r}")

        micro_ids, micro_gen, _ = run_micro(prompt, N_TOKENS)
        print(f"micro prompt_ids: {micro_ids}")
        print(f"micro gen ({len(micro_gen)}): {micro_gen}")
        print(f"micro text: {tok.decode(micro_gen)!r}")

        n = min(len(hf_gen), len(micro_gen))
        match = hf_gen[:n] == micro_gen[:n]
        ok_all = ok_all and match
        print(f"prompt_ids match: {list(hf_ids) == list(micro_ids)}")
        print(f"gen match (first {n}): {match}")
        if not match:
            for i in range(n):
                mark = " " if hf_gen[i] == micro_gen[i] else "X"
                print(f"  [{i}] {mark} hf={hf_gen[i]} micro={micro_gen[i]}")
    print(f"\n{'ALL MATCH' if ok_all else 'MISMATCH'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
