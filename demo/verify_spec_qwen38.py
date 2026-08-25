"""Qwen3.8-27B W8A16 投机解码正确性验证：spec_decode vs 非 spec greedy 逐 token 对比。

贪心投机解码等价性：单序列下 spec 输出应与非 spec greedy 输出逐 token 一致。

用法：CUDA_VISIBLE_DEVICES=4 MICRO_W8A16=1 python3 demo/verify_spec_qwen38.py
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
OUT_TOK = int(os.environ.get("OUT_TOK", "64"))
N_SPEC = int(os.environ.get("N_SPEC", "7"))


def run_non_spec(eng, prompt, out_tok):
    """非 spec greedy：走 engine 正常 add_request + step 循环，返回 output_ids。"""
    eng.add_request(prompt, out_tok, temperature=0.0, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    for seq in list(eng.scheduler.finished_sequences):
        return list(seq.output_ids)
    return []


def main():
    print(f"model={MODEL}\ndraft={DRAFT}\nout={OUT_TOK} N_spec={N_SPEC}")
    eng = InferenceEngine(
        MODEL, max_batch_size=16, max_prefill_tokens=4096,
        spec_decode=True, draft_model_path=DRAFT,
        num_speculative_tokens=N_SPEC)

    # ---- 非 spec greedy ----
    t0 = time.perf_counter()
    non_spec = run_non_spec(eng, PROMPT, OUT_TOK)
    t_ns = time.perf_counter() - t0
    print(f"\n[非 spec] {len(non_spec)} tokens in {t_ns:.2f}s = {len(non_spec)/t_ns:.1f} tok/s")

    # ---- spec decode（先 warmup 一次丢弃，摊销 TileLang GEMM 一次性编译）----
    _ = eng.generate_spec_decode(PROMPT, 16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = eng.generate_spec_decode(PROMPT, OUT_TOK)
    t_sp = time.perf_counter() - t0
    spec = res["tokens"]
    print(f"[spec]     {len(spec)} tokens in {t_sp:.2f}s = {res['tok_s']:.1f} tok/s "
          f"acceptance={res['avg_acceptance']:.3f} steps={res['num_steps']}")

    # ---- 对比 ----
    n = min(len(non_spec), len(spec))
    match = non_spec[:n] == spec[:n]
    print(f"\n对比: 非spec={len(non_spec)} spec={len(spec)} 前{n}个逐token一致={match}")
    if not match:
        for i in range(n):
            if non_spec[i] != spec[i]:
                print(f"  首个分歧 @ {i}: 非spec={non_spec[i]} spec={spec[i]}")
                print(f"  非spec[{i}:{i+8}]={non_spec[i:i+8]}")
                print(f"  spec  [{i}:{i+8}]={spec[i:i+8]}")
                break
    print("\n非spec文本:", eng.tokenizer.decode(non_spec, skip_special_tokens=True)[:300])
    print("spec文本:  ", eng.tokenizer.decode(spec, skip_special_tokens=True)[:300])
    print(f"\n{'PASS' if match else 'FAIL'}: spec 与非 spec greedy 逐 token 一致")


if __name__ == "__main__":
    main()
