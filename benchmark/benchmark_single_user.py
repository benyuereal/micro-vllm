"""单用户 decode 基准：micro-vllm vs vLLM 0.21.0 vs nano-vllm。

测什么：
  - bs=1、长上下文（256 in / 768 out，合计 1024）、temp=0.01
  - 单请求 wall time，吞吐 tok/s = 768 / wall（含 prefill，口径三者一致）
  - 768 个 decode step 放大 per-step 开销差异；KV 增长到满 1024，
    暴露 attention 实现 / paged KV 差异
  - 每框架 7 轮取中位数，独立进程跑，GPU 由调用方指定

用法：
  python3 benchmark_single_user.py <micro|vllm|nano>
  例: python3 benchmark_single_user.py micro

依赖：
  - 空闲 GPU（CUDA_VISIBLE_DEVICES 由调用方指定，如 CUDA_VISIBLE_DEVICES=1）
  - micro 路径自动定位为本仓库根目录（脚本所在目录的上一级）
  - nano-vllm 路径默认 /models/nano-vllm，可用环境变量 NANO_VLLM_PATH 覆盖
  - 模型路径默认 /models/Qwen3-0.6B，可用环境变量 MODEL_PATH 覆盖
"""
import os, sys, time, statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

MODE = sys.argv[1]
# micro graph 路径固定 1024 上下文（>1024 留待 tile op 恢复），故 IN+OUT ≤ 1024。
# 取 256 in / 768 out（合计 1024）：768 个 decode step（放大 per-step 开销差异），
# KV 增长到满 1024（暴露 attention / paged KV 差异）。
# 注：prefill=256 恰为 block_size 整数倍，曾触发 cache_manager.alloc 的 off-by-one
# （_pos 误置 0 → 首 decode 越界），已修复（last_pos = n%bs or bs）。
IN_TOK = 256
OUT_TOK = 768
TEMP = 0.01
ROUNDS = 7


def make_prompt():
    """构造恰好 IN_TOK token 的 prompt（用模型自带 tokenizer 对齐）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < IN_TOK:
        ids += tok.encode(" data")
    ids = ids[:IN_TOK]
    return tok.decode(ids)


def run_micro():
    sys.path.insert(0, REPO_ROOT)
    import torch
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    # IN+OUT=1024 恰好落在 graph 固定 1024 上下文内，用默认 max_context_length。
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    prompt = make_prompt()
    # warmup（触发 graph 捕获）
    eng.add_request(prompt, 8, temperature=TEMP, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    walls = []
    for _ in range(ROUNDS):
        eng.add_request(prompt, OUT_TOK, temperature=TEMP, top_p=1.0)
        torch.cuda.synchronize(); t0 = time.time()
        while True:
            b, bt = eng.get_next_batch()
            if not b: break
            ctx = BatchInferenceContext(len(b), bt, b)
            eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        torch.cuda.synchronize(); walls.append(time.time() - t0)
    return walls


def run_vllm():
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt()
    llm.generate([prompt], SamplingParams(temperature=TEMP, max_tokens=8))
    walls = []
    for _ in range(ROUNDS):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=TEMP, ignore_eos=True,
                                              max_tokens=OUT_TOK), use_tqdm=False)
        walls.append(time.time() - t0)
    return walls


def run_nano():
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt()
    llm.generate([prompt], SamplingParams(temperature=TEMP, max_tokens=8))
    walls = []
    for _ in range(ROUNDS):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=TEMP, ignore_eos=True,
                                              max_tokens=OUT_TOK), use_tqdm=False)
        walls.append(time.time() - t0)
    return walls


def main():
    if MODE == "micro":
        walls = run_micro()
    elif MODE == "vllm":
        walls = run_vllm()
    elif MODE == "nano":
        walls = run_nano()
    else:
        raise SystemExit(f"unknown mode: {MODE} (micro|vllm|nano)")

    med_wall = statistics.median(walls)
    print(f"{MODE} bs=1 ({IN_TOK}in/{OUT_TOK}out): {OUT_TOK/med_wall:.1f} tok/s | "
          f"wall {med_wall*1000:.0f} ms | rounds={[f'{w*1000:.0f}' for w in walls]}")


if __name__ == "__main__":
    # vLLM 用 multiprocessing spawn 起 engine core 子进程，会重新 import 本模块；
    # 无 __main__ guard 时子进程会重跑 run_vllm() 导致递归 spawn 崩溃。
    main()
