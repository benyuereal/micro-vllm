"""Qwen3.8-27B W8A16 greedy 对齐测试：micro-vllm vs HF transformers。

HF 参考（/tmp/hf_ref_27b.json）由 hf_ref_27b.py 在 CPU 上预生成（bf16 54G 超 GPU4 45G，
HF 无法在 GPU4 直接跑；in-place 反量化 pack-quantized int8 → bf16 后 generate）。
本脚本只跑 micro（GPU4），逐 token 对比预生成的 HF 参考。

用法：
  CUDA_VISIBLE_DEVICES=4 python3 demo/align_qwen38.py
"""
import os, sys, json
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
REF = os.environ.get("HF_REF", "/tmp/hf_ref_27b.json")
N_TOKENS = int(os.environ.get("N_TOKENS", "24"))


def run_micro_all(prompts, n):
    """一个 engine 跑所有 prompt（27B int8 30G 常驻，重载会 OOM；GDN 状态池是类级
    单例跨 engine 持久，单 engine 顺序跑最省显存）。返回 {prompt: (ids, gen, tok)}。"""
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    # max_batch_size=16：GDN 状态池 = 16 × 48 层 × 48×128×128×4B ≈ 2.4GB（64 会 9.6GB，
    # 27B int8 30G + 状态池超 GPU4 45G OOM）。单用户对齐只需 1。
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096)
    for p in prompts:
        eng.add_request(p, n, temperature=0.0, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    tok = eng.tokenizer
    out = {}
    for p in prompts:
        prompt_ids = tok.encode(p, add_special_tokens=True)
        for seq in list(eng.scheduler.finished_sequences) + list(eng.scheduler.running_sequences):
            if list(seq.input_ids) == prompt_ids:
                out[p] = (prompt_ids, list(seq.output_ids), tok)
                break
        else:
            raise RuntimeError(f"micro: 找不到 prompt {p!r} 的 seq")
    return out


def main():
    ref = json.load(open(REF))
    prompts = list(ref.keys())
    micro = run_micro_all(prompts, N_TOKENS)
    ok_all = True
    for prompt in prompts:
        r = ref[prompt]
        print(f"\n===== prompt: {prompt!r} =====")
        hf_ids = r["prompt_ids"]
        hf_gen = r["gen_tokens"]
        print(f"HF prompt_ids: {hf_ids}")
        print(f"HF gen ({len(hf_gen)}): {hf_gen}")
        print(f"HF text: {r['text']!r}")

        micro_ids, micro_gen, tok = micro[prompt]
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
