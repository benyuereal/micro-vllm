"""prefix cache 长 prefill 验证：共享 768 前缀 + 128 后缀（总 896 ≤ 1024 graph 上限）。

命中后 prefill 896→128 token（省 86% 计算量）。长 prefill 下计算量随 token 数
线性（GEMM）+ 平方（attention），固定开销占比下降，wall 收益应显著。

用法：CUDA_VISIBLE_DEVICES=2 python3 benchmark/test_prefix_cache_long.py
"""
import os, sys, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

PREFIX_TOK = 768   # 共享前缀（3 个满块）
SUFFIX_TOK = 128   # 独有后缀
OUT_TOK = 32
TEMP = 0.0


def make_prompt(tok, prefix_ids, suffix_seed):
    suffix = tok.encode(f" Question {suffix_seed}: what is the answer?")
    ids = prefix_ids + suffix
    while len(ids) < PREFIX_TOK + SUFFIX_TOK:
        ids += tok.encode(" more")
    ids = ids[:PREFIX_TOK + SUFFIX_TOK]
    return tok.decode(ids)


def run_request(eng, prompt, out_tok):
    import torch
    from core.inference_context import BatchInferenceContext
    eng.add_request(prompt, out_tok, temperature=TEMP, top_p=1.0)
    torch.cuda.synchronize()
    t0 = time.time()
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    torch.cuda.synchronize()
    wall = time.time() - t0
    for seq, out_ids in eng.scheduler.get_finished_results():
        return wall, out_ids
    return wall, None


def main():
    import torch
    from transformers import AutoTokenizer
    from core.engine import InferenceEngine

    tok = AutoTokenizer.from_pretrained(MODEL)
    prefix_ids = tok.encode("You are a helpful assistant. " * 40)[:PREFIX_TOK]
    prompt_a = make_prompt(tok, prefix_ids, "A")
    prompt_b = make_prompt(tok, prefix_ids, "B")

    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096)
    run_request(eng, tok.decode(tok.encode("warmup")[:16]), 4)

    # 1. 基线（无缓存）：prefill 896
    wall_b, out_b = run_request(eng, prompt_b, OUT_TOK)
    n_reg = len(eng.cache_manager._prefix_cache)
    print(f"[1] 基线: wall {wall_b*1000:.1f}ms, 登记前缀块 {n_reg}（应 {PREFIX_TOK//256}）")

    # 2. 命中：prefill 128
    wall_a, out_a = run_request(eng, prompt_a, OUT_TOK)
    print(f"[2] 命中: wall {wall_a*1000:.1f}ms")

    # 3. 同 prompt 再命中（正确性对照）
    wall_b2, out_b2 = run_request(eng, prompt_b, OUT_TOK)
    print(f"[3] 命中(同B): wall {wall_b2*1000:.1f}ms")

    ok = out_b == out_b2
    print(f"\n正确性: O0 == O1 → {ok}")
    if not ok:
        for i, (x, y) in enumerate(zip(out_b, out_b2)):
            if x != y:
                print(f"  首个分叉 @tok {i}: base={x} cached={y}")
                break

    save = wall_b - wall_b2
    print(f"\n性能: 基线 {wall_b*1000:.1f}ms → 命中 {wall_b2*1000:.1f}ms, "
          f"省 {save*1000:.1f}ms ({save/wall_b*100:.1f}%)")
    print(f"  (prefill 896→128 token，省 86% 计算量)")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
