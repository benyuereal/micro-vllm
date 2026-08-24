"""TP 性能对比：micro-vllm vs vLLM 0.21.0，TP=1/2。

单用户 decode（256 in / 768 out，7 轮取中位数）+ 多用户吞吐（bs=32/64 连续批处理）。

用法：
  micro TP=1 单用户:  CUDA_VISIBLE_DEVICES=2 python3 benchmark/benchmark_tp_perf.py micro 1 single
  micro TP=2 单用户:  CUDA_VISIBLE_DEVICES=2,5 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
                        --master_port=29520 benchmark/benchmark_tp_perf.py micro 2 single
  micro TP=2 多用户:  CUDA_VISIBLE_DEVICES=2,5 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
                        --master_port=29521 benchmark/benchmark_tp_perf.py micro 2 multi 32
  vllm  TP=1 单用户:  CUDA_VISIBLE_DEVICES=2 python3 benchmark/benchmark_tp_perf.py vllm 1 single
  vllm  TP=2 单用户:  CUDA_VISIBLE_DEVICES=2,5 python3 benchmark/benchmark_tp_perf.py vllm 2 single
  vllm  TP=2 多用户:  CUDA_VISIBLE_DEVICES=2,5 python3 benchmark/benchmark_tp_perf.py vllm 2 multi 32

口径对齐 benchmark_single_user.py / benchmark_throuput.py：
  - 单用户：256 in / 768 out，temp=0.01，7 轮中位数，tok/s = 768/wall（含 prefill）
  - 多用户：N 条 128 in / 256 out（ignore_eos 跑满），聚合 tok/s = N*256/wall
micro TP=2 走 torchrun（rank0 驱动 + 计时，非 rank0 receive 同步，对齐 api_server）。
"""
import os, sys, time, statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

MODE = sys.argv[1]          # micro | vllm
TP = int(sys.argv[2])       # 1 | 2
KIND = sys.argv[3] if len(sys.argv) > 3 else "single"   # single | multi
BS = int(sys.argv[4]) if len(sys.argv) > 4 else 32

# 单用户口径（对齐 benchmark_single_user.py）
S_IN, S_OUT, S_ROUNDS, S_TEMP = 256, 768, 7, 0.01
# 多用户口径（对齐 benchmark_throuput.py）
M_IN, M_OUT, M_TEMP = 128, 256, 0.01


def make_prompt(n_tok):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < n_tok:
        ids += tok.encode(" data")
    ids = ids[:n_tok]
    return tok.decode(ids)


# ----------------------------- micro -----------------------------
def run_micro_single():
    import torch
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core.parallel_config import get_rank, rank0
    import torch.distributed as dist

    rank = get_rank()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    prompt = make_prompt(S_IN)

    # warmup（触发 graph 捕获 + 算法选择）
    if rank0():
        eng.add_request(prompt, 8, temperature=S_TEMP, top_p=1.0)
    if ws > 1:
        _micro_loop(eng, max_steps=8)
    else:
        _micro_loop(eng, max_steps=8)

    walls = []
    for _ in range(S_ROUNDS):
        if rank0():
            eng.add_request(prompt, S_OUT, temperature=S_TEMP, top_p=1.0)
        torch.cuda.synchronize()
        t0 = time.time()
        _micro_loop(eng, max_steps=S_OUT + 8)
        torch.cuda.synchronize()
        if rank0():
            walls.append(time.time() - t0)

    if rank0():
        med = statistics.median(walls)
        print(f"micro TP={TP} single bs=1 ({S_IN}in/{S_OUT}out): {S_OUT/med:.1f} tok/s | "
              f"wall {med*1000:.0f} ms | rounds={[f'{w*1000:.0f}' for w in walls]}")
        print(f"micro TP={TP} single mem_mb={torch.cuda.memory_allocated()/(1<<20):.0f}")


def _micro_loop(eng, max_steps):
    """rank0 驱动 + 非 rank0 同步的推理循环，跑完当前所有请求（或 max_steps 步）。"""
    import torch
    import torch.distributed as dist
    from core.inference_context import BatchInferenceContext
    from core.parallel_config import get_rank, rank0

    rank = get_rank()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    steps = 0
    while steps < max_steps:
        if ws > 1:
            if rank0():
                b, bt = eng.get_next_batch()
                done = (not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue)
                if bt == "waiting" or not b:
                    BatchInferenceContext(0, "waiting").broadcast()
                else:
                    ctx = BatchInferenceContext(len(b), bt, b)
                    ctx.broadcast()
                    eng.step(ctx)
                    eng.collect(ctx)
                    ctx.broadcast()
                    eng.update_sequences(ctx.sequences)
                dt = torch.tensor([1 if done else 0], device=eng.device)
                dist.broadcast(dt, src=0)
            else:
                ctx = BatchInferenceContext.receive(eng.tokenizer)
                if ctx.batch_type != "waiting" and ctx.batch_size > 0:
                    eng.step(ctx)
                    ctx = BatchInferenceContext.receive(eng.tokenizer)
                    eng.update_sequences(ctx.sequences)
                dt = torch.zeros(1, device=eng.device)
                dist.broadcast(dt, src=0)
            if int(dt.item()) == 1:
                break
        else:
            b, bt = eng.get_next_batch()
            if not b:
                break
            ctx = BatchInferenceContext(len(b), bt, b)
            eng.step(ctx)
            eng.collect(ctx)
            eng.update_sequences(ctx.sequences)
        steps += 1


def run_micro_multi():
    import torch
    from core.engine import InferenceEngine
    from core.parallel_config import get_rank, rank0

    rank = get_rank()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    eng = InferenceEngine(MODEL, max_batch_size=max(BS, 64), max_prefill_tokens=4096)
    prompt = make_prompt(M_IN)
    # warmup
    if rank0():
        eng.add_request(prompt, 8, temperature=M_TEMP, top_p=1.0)
    _micro_loop(eng, max_steps=8)

    if rank0():
        for _ in range(BS):
            eng.add_request(prompt, M_OUT, temperature=M_TEMP, top_p=1.0)
    torch.cuda.synchronize()
    t0 = time.time()
    n_tok = 0
    # 统计真实 token 数：rank0 从 scheduler 完成结果累加
    _micro_loop(eng, max_steps=BS * M_OUT + 64)
    torch.cuda.synchronize()
    dt = time.time() - t0
    if rank0():
        # 完成 token 数 = BS * M_OUT（ignore_eos 语义：跑满 M_OUT）
        total = BS * M_OUT
        print(f"micro TP={TP} multi bs={BS}: {total/dt:.1f} tok/s ({total} tok, {dt:.3f}s)")
        print(f"micro TP={TP} multi mem_mb={torch.cuda.memory_allocated()/(1<<20):.0f}")


# ----------------------------- vllm -----------------------------
def run_vllm_single():
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=TP, max_model_len=4096)
    prompt = make_prompt(S_IN)
    llm.generate([prompt], SamplingParams(temperature=S_TEMP, max_tokens=8))
    walls = []
    for _ in range(S_ROUNDS):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=S_TEMP, ignore_eos=True,
                                              max_tokens=S_OUT), use_tqdm=False)
        walls.append(time.time() - t0)
    med = statistics.median(walls)
    print(f"vllm TP={TP} single bs=1 ({S_IN}in/{S_OUT}out): {S_OUT/med:.1f} tok/s | "
          f"wall {med*1000:.0f} ms | rounds={[f'{w*1000:.0f}' for w in walls]}")


def run_vllm_multi():
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=TP, max_model_len=4096)
    prompt = make_prompt(M_IN)
    llm.generate([prompt], SamplingParams(temperature=M_TEMP, max_tokens=8))
    prompts = [prompt] * BS
    sps = [SamplingParams(temperature=M_TEMP, ignore_eos=True, max_tokens=M_OUT) for _ in range(BS)]
    t0 = time.time()
    llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    total = BS * M_OUT
    print(f"vllm TP={TP} multi bs={BS}: {total/dt:.1f} tok/s ({total} tok, {dt:.3f}s)")


if MODE == "micro":
    if KIND == "single":
        run_micro_single()
    else:
        run_micro_multi()
elif MODE == "vllm":
    if KIND == "single":
        run_vllm_single()
    else:
        run_vllm_multi()
else:
    raise SystemExit(f"unknown mode: {MODE} (micro|vllm)")
