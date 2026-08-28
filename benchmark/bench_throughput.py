"""统一吞吐基准：单用户长上下文 + N 请求连续批处理，一个参数化入口。

合并自 benchmark_single_user.py（单用户）与 benchmark1000_throughput.py（批量）。

两种模式（由 --n 决定）：
  --n 1   单用户模式：bs=1、长上下文（默认 256 in / 768 out，合计 1024）、temp=0.01，
          单请求 wall time，吞吐 tok/s = out / wall（含 prefill，口径三者一致）；
          默认 7 轮取中位数，独立进程跑，GPU 由调用方指定。
  --n N   批量模式：N 个请求全入队再排空，聚合 decode 吞吐 tok/s + req/s + 完成数；
          temp=0.6，ignore_eos 跑满随机 max_tokens（[max-tok/2, max-tok]）以测纯吞吐。

后端（--backend）：
  micro  InferenceEngine 直接驱动（BatchInferenceContext 循环）
  vllm   vllm.LLM（仅单用户模式支持）
  nano   nanovllm（路径 NANO_VLLM_PATH 环境变量）
  all    依次跑该模式支持的全部后端（每个后端独立子进程，避免相互干扰）

用法：
  # 单用户长上下文（原 benchmark_single_user 语义）
  python3 benchmark/bench_throughput.py --n 1 --in-tok 256 --out-tok 768 --rounds 7 --backend micro
  python3 benchmark/bench_throughput.py --n 1 --backend all

  # N 请求连续批处理（原 benchmark1000 语义）
  python3 benchmark/bench_throughput.py --n 1000 --max-tok 80 --backend micro
  python3 benchmark/bench_throughput.py --n 1000 --max-tok 80 --backend all

环境变量（可选）：
  MODEL_PATH      默认 /models/Qwen3.8-27B-INT8-W8A16-MTP
  NANO_VLLM_PATH  默认 /models/nano-vllm
GPU 由调用方通过 CUDA_VISIBLE_DEVICES 指定（勿用 GPU0，那是常驻 DeepSeek 服务）。
"""
import os, sys, time, random, statistics, argparse, subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")

TEMP_SINGLE = 0.01   # 单用户模式
TEMP_BATCH = 0.6     # 批量模式

# 批量模式 15 个中文 prompt 池
PROMPTS = [
    "写一个 SpringBoot 文件上传代码",
    "解释区块链的共识机制",
    "用JavaScript实现一个Promise限流池",
    "写一篇关于元宇宙未来的短文",
    "如何学习网络安全？给出学习路径",
    "比较SQL和NoSQL数据库的优缺点",
    "写一个关于时间旅行的科幻故事开头",
    "用Rust实现一个简单的链表结构",
    "写一篇关于远程工作利弊的分析",
    "如何成为一名全栈开发者？",
    "比较微服务和单体架构的优缺点",
    "解释机器学习中的过拟合与欠拟合",
    "用Python实现一个简单的神经网络",
    "如何系统地学习数据结构与算法？",
    "解释什么是碳中和以及实现路径",
]


def make_prompt(in_tok):
    """构造恰好 in_tok token 的 prompt（用模型自带 tokenizer 对齐）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < in_tok:
        ids += tok.encode(" data")
    ids = ids[:in_tok]
    return tok.decode(ids)


# ---------------------------------------------------------------- 单用户模式

def run_single_micro(in_tok, out_tok, rounds, max_batch):
    sys.path.insert(0, REPO_ROOT)
    import torch
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    # IN+OUT=1024 恰好落在 graph 固定 1024 上下文内，用默认 max_context_length。
    # max_batch_size 决定 GDN 状态池大小（类级单例，pool=max_bs）：27B 在 44GiB 卡上
    # 64 会 OOM（池 9GiB + 权重/KV ~37GiB），单用户 bs=1 用小值即可。
    eng = InferenceEngine(MODEL, max_batch_size=max_batch, max_prefill_tokens=4096)
    prompt = make_prompt(in_tok)
    # warmup（触发 graph 捕获）
    eng.add_request(prompt, 8, temperature=TEMP_SINGLE, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    walls = []
    for _ in range(rounds):
        eng.add_request(prompt, out_tok, temperature=TEMP_SINGLE, top_p=1.0)
        torch.cuda.synchronize(); t0 = time.time()
        while True:
            b, bt = eng.get_next_batch()
            if not b: break
            ctx = BatchInferenceContext(len(b), bt, b)
            eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        torch.cuda.synchronize(); walls.append(time.time() - t0)
    return walls


def run_single_vllm(in_tok, out_tok, rounds):
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt(in_tok)
    llm.generate([prompt], SamplingParams(temperature=TEMP_SINGLE, max_tokens=8))
    walls = []
    for _ in range(rounds):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=TEMP_SINGLE, ignore_eos=True,
                                              max_tokens=out_tok), use_tqdm=False)
        walls.append(time.time() - t0)
    return walls


def run_single_nano(in_tok, out_tok, rounds):
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt(in_tok)
    llm.generate([prompt], SamplingParams(temperature=TEMP_SINGLE, max_tokens=8))
    walls = []
    for _ in range(rounds):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=TEMP_SINGLE, ignore_eos=True,
                                              max_tokens=out_tok), use_tqdm=False)
        walls.append(time.time() - t0)
    return walls


def run_single(backend, in_tok, out_tok, rounds, max_batch):
    if backend == "micro":
        walls = run_single_micro(in_tok, out_tok, rounds, max_batch)
    elif backend == "vllm":
        walls = run_single_vllm(in_tok, out_tok, rounds)
    elif backend == "nano":
        walls = run_single_nano(in_tok, out_tok, rounds)
    else:
        raise SystemExit(f"unknown backend: {backend} (micro|vllm|nano)")
    med_wall = statistics.median(walls)
    print(f"{backend} bs=1 ({in_tok}in/{out_tok}out): {out_tok/med_wall:.1f} tok/s | "
          f"wall {med_wall*1000:.0f} ms | rounds={[f'{w*1000:.0f}' for w in walls]}")


# ---------------------------------------------------------------- 批量模式

def run_batch_micro(total, maxtok, prompts_list, maxtoks_list, max_batch):
    sys.path.insert(0, REPO_ROOT)
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    # max_batch_size 决定 GDN 状态池大小（类级单例，pool=max_bs）：27B 在 44GiB 卡上
    # 512 会 OOM（池 ~70GiB），用 --max-batch 控制（默认 16，够 bs≤16 并发）。
    eng = InferenceEngine(MODEL, max_batch_size=max_batch, max_prefill_tokens=16384, max_context_length=1024)
    for i in range(total):
        eng.add_request(prompts_list[i], maxtoks_list[i], temperature=TEMP_BATCH, top_p=1.0)
    eng.scheduler.finished_sequences.clear()
    t0 = time.time()
    steps = 0
    while True:
        batch, batch_type = eng.get_next_batch()
        if batch_type == "idle" or not batch:
            if not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue:
                break
            continue
        ctx = BatchInferenceContext(len(batch), batch_type, batch)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        steps += 1
    wall = time.time() - t0
    out_tok = sum(len(s.output_ids) for s in eng.scheduler.finished_sequences)
    n_done = len(eng.scheduler.finished_sequences)
    return wall, out_tok, n_done, steps


def run_batch_nano(total, maxtok, prompts_list, maxtoks_list):
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=1024)
    sps = [SamplingParams(temperature=TEMP_BATCH, ignore_eos=True, max_tokens=mt) for mt in maxtoks_list]
    llm.generate(["warmup"], SamplingParams(temperature=TEMP_BATCH, ignore_eos=True, max_tokens=2), use_tqdm=False)
    t0 = time.time()
    outputs = llm.generate(prompts_list, sps, use_tqdm=False)
    wall = time.time() - t0
    out_tok = sum(len(o["token_ids"]) for o in outputs)
    return wall, out_tok, len(outputs)


def run_batch(backend, total, maxtok, max_batch):
    prompts_list = [PROMPTS[i % len(PROMPTS)] for i in range(total)]
    maxtoks_list = [random.randint(maxtok // 2, maxtok) for _ in range(total)]
    if backend == "micro":
        w, ot, nd, st = run_batch_micro(total, maxtok, prompts_list, maxtoks_list, max_batch)
        print(f"=== micro-vllm   TOTAL={total} maxtok~{maxtok} ===")
        print(f"  完成: {nd}/{total}  wall={w:.2f}s  steps={st}")
        print(f"  输出 token 总数: {ot}")
        print(f"  聚合 decode 吞吐: {ot/w:.1f} tok/s")
        print(f"  请求速率: {total/w:.1f} req/s")
    elif backend == "nano":
        w, ot, nd = run_batch_nano(total, maxtok, prompts_list, maxtoks_list)
        print(f"=== nano-vllm    TOTAL={total} maxtok~{maxtok} ===")
        print(f"  完成: {nd}/{total}  wall={w:.2f}s")
        print(f"  输出 token 总数: {ot}")
        print(f"  聚合 decode 吞吐: {ot/w:.1f} tok/s")
        print(f"  请求速率: {total/w:.1f} req/s")
    else:
        raise SystemExit(f"unknown backend: {backend} (批量模式支持 micro|nano)")
    print()


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="micro-vllm 统一吞吐基准（单用户 / N 请求批量）")
    ap.add_argument("--n", type=int, default=1,
                    help="1=单用户模式（多轮中位数）；N>1=批量模式（N 请求全入队排空）")
    ap.add_argument("--in-tok", type=int, default=256, help="单用户模式输入 token 数（默认 256）")
    ap.add_argument("--out-tok", type=int, default=768, help="单用户模式输出 token 数（默认 768）")
    ap.add_argument("--rounds", type=int, default=7, help="单用户模式轮数，取中位数（默认 7）")
    ap.add_argument("--max-tok", type=int, default=80,
                    help="批量模式随机 max_tokens 上限，实际取 [max-tok/2, max-tok]（默认 80）")
    ap.add_argument("--max-batch", type=int, default=16,
                    help="micro 引擎 max_batch_size（决定 GDN 状态池大小，27B 在 44GiB 卡上"
                         " 64/512 会 OOM；默认 16，单用户 bs=1 足够，批量模式按并发需求调大）")
    ap.add_argument("--backend", default="micro",
                    help="micro|vllm|nano|all（vllm 仅单用户模式；all=该模式支持的全部后端，独立子进程）")
    args = ap.parse_args()

    if args.n < 1:
        raise SystemExit("--n 必须 >= 1")

    if args.n == 1:
        backends = ["micro", "vllm", "nano"] if args.backend == "all" else [args.backend]
        for b in backends:
            if b not in ("micro", "vllm", "nano"):
                raise SystemExit(f"unknown backend: {b} (micro|vllm|nano|all)")
            if len(backends) == 1:
                run_single(b, args.in_tok, args.out_tok, args.rounds, args.max_batch)
            else:
                # 独立子进程跑，避免多框架同进程相互干扰（原脚本语义）
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--n", "1", "--in-tok", str(args.in_tok), "--out-tok", str(args.out_tok),
                       "--rounds", str(args.rounds), "--max-batch", str(args.max_batch),
                       "--backend", b]
                subprocess.run(cmd, check=True)
    else:
        backends = ["micro", "nano"] if args.backend == "all" else [args.backend]
        for b in backends:
            if b not in ("micro", "nano"):
                raise SystemExit(f"批量模式不支持 backend: {b}（支持 micro|nano|all）")
            if len(backends) == 1:
                run_batch(b, args.n, args.max_tok, args.max_batch)
            else:
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--n", str(args.n), "--max-tok", str(args.max_tok),
                       "--max-batch", str(args.max_batch), "--backend", b]
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # vLLM 用 multiprocessing spawn 起 engine core 子进程，会重新 import 本模块；
    # 无 __main__ guard 时子进程会重跑 main() 导致递归 spawn 崩溃。
    import torch
    torch.manual_seed(0)
    random.seed(0)
    main()
