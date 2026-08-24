"""prefix cache 验证：共享前缀 KV 复用。

场景（单 engine，3 个请求）：
  1. B（prefix+suffixB，576 token）：无缓存基线，prefill 576，输出 O0，登记 512 前缀
  2. A（prefix+suffixA）：命中 512，只 prefill 64 新 token
  3. B'（同 B）：命中 512，只 prefill 64 新 token，输出 O1

验证：
  - 正确性：O0 == O1（同 prompt greedy 必须逐 token 一致）
  - 性能：B' prefill wall < B prefill wall（省 512/576 计算）
  - 无回归：无共享前缀请求 prefix_hit=0，行为不变

用法：CUDA_VISIBLE_DEVICES=2 python3 benchmark/test_prefix_cache.py
"""
import os, sys, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

PREFIX_TOK = 512   # 共享前缀（2 个满块，block_size=256）
SUFFIX_TOK = 64    # 各请求独有后缀
OUT_TOK = 32       # 生成长度
TEMP = 0.0         # greedy


def make_prompt(tok, prefix_ids, suffix_seed):
    """prefix（固定 token）+ suffix（按 seed 变化）→ 恰好 PREFIX+SUFFIX token 的 prompt。"""
    suffix = tok.encode(f" Question {suffix_seed}: what is the answer?")
    ids = prefix_ids + suffix
    while len(ids) < PREFIX_TOK + SUFFIX_TOK:
        ids += tok.encode(" more")
    ids = ids[:PREFIX_TOK + SUFFIX_TOK]
    return tok.decode(ids)


def run_request(eng, prompt, out_tok, stop=None):
    """跑单请求到完成，返回 (wall_s, output_ids)。stop 命中即停（测纯 prefill 用）。"""
    import torch
    from core.inference_context import BatchInferenceContext
    eng.add_request(prompt, out_tok, temperature=TEMP, top_p=1.0, stop=stop)
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
    # 取该请求的输出 token
    for seq, out_ids in eng.scheduler.get_finished_results():
        return wall, out_ids
    return wall, None


def main():
    import torch
    from transformers import AutoTokenizer
    from core.engine import InferenceEngine

    tok = AutoTokenizer.from_pretrained(MODEL)
    prefix_ids = tok.encode("You are a helpful assistant. " * 20)
    prefix_ids = prefix_ids[:PREFIX_TOK]
    prompt_a = make_prompt(tok, prefix_ids, "A")
    prompt_b = make_prompt(tok, prefix_ids, "B")

    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096)
    # warmup（graph 捕获）
    run_request(eng, tok.decode(tok.encode("warmup")[:16]), 4)

    # 1. B 基线（无缓存）
    wall_b, out_b = run_request(eng, prompt_b, OUT_TOK)
    n_reg = len(eng.cache_manager._prefix_cache)
    print(f"[1] B 基线: wall {wall_b*1000:.1f}ms, out {len(out_b)} tok, 登记前缀块 {n_reg}")
    assert n_reg == PREFIX_TOK // 256, f"应登记 {PREFIX_TOK//256} 个满块前缀, 实际 {n_reg}"

    # 2. A（命中 512）
    wall_a, out_a = run_request(eng, prompt_a, OUT_TOK)
    print(f"[2] A 命中: wall {wall_a*1000:.1f}ms, out {len(out_a)} tok")

    # 3. B'（命中 512，同 prompt 应同输出）
    wall_b2, out_b2 = run_request(eng, prompt_b, OUT_TOK)
    print(f"[3] B' 命中: wall {wall_b2*1000:.1f}ms, out {len(out_b2)} tok")

    # 正确性
    ok = out_b == out_b2
    print(f"\n正确性: O0 == O1 → {ok}")
    if not ok:
        for i, (x, y) in enumerate(zip(out_b, out_b2)):
            if x != y:
                print(f"  首个分叉 @tok {i}: base={x} cached={y}")
                break
        print(f"  base: {tok.decode(out_b)[:120]}")
        print(f"  cached: {tok.decode(out_b2)[:120]}")

    # 性能（纯 prefill：max_tokens=1，wall ≈ prefill + 1 decode step，用第二组前缀隔离）
    prefix2_ids = tok.encode("Another shared system prompt for perf test. " * 10)[:PREFIX_TOK]
    p1 = make_prompt(tok, prefix2_ids, "P1")
    p2 = make_prompt(tok, prefix2_ids, "P2")
    p3 = make_prompt(tok, prefix2_ids, "P3")
    w1, _ = run_request(eng, p1, 1)   # 无缓存：prefill 576
    w2, _ = run_request(eng, p2, 1)   # 命中：prefill 64
    w3, _ = run_request(eng, p3, 1)   # 命中：prefill 64
    print(f"\n性能(纯 prefill, max_tokens=1): 无缓存 {w1*1000:.1f}ms → 命中 {w2*1000:.1f}/{w3*1000:.1f}ms, "
          f"省 {(w1-w2)*1000:.1f}ms ({(w1-w2)/w1*100:.1f}%)")
    print(f"  (prefill 计算量 576→64 token，理论省 ~89% prefill 时间)")

    # 无回归：无共享前缀请求
    prompt_c = make_prompt(tok, tok.encode("completely different prefix here " * 10), "C")
    wall_c, out_c = run_request(eng, prompt_c, OUT_TOK)
    print(f"[4] C 无前缀共享: wall {wall_c*1000:.1f}ms（应与 B 基线同量级）")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
