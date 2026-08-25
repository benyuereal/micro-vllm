"""TP 性能对比：micro-vllm (torchrun) vs vLLM (tensor_parallel_size)。

单用户 decode：256 in / 768 out，temp=0.01，7 轮取中位数（对齐 benchmark_single_user.py 口径）。
多用户吞吐：N 条 128 in / 256 out（ignore_eos 跑满），聚合 tok/s = N*256/wall。

用法：
  micro TP=1:  CUDA_VISIBLE_DEVICES=2 python3 benchmark/benchmark_tp.py micro 1
  micro TP=2:  CUDA_VISIBLE_DEVICES=2,5 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
                --master_port=29602 benchmark/benchmark_tp.py micro 2
  vllm  TP=1:  CUDA_VISIBLE_DEVICES=2 python3 benchmark/benchmark_tp.py vllm 1
  vllm  TP=2:  CUDA_VISIBLE_DEVICES=2,5 python3 benchmark/benchmark_tp.py vllm 2

  多用户：追加 N 参数，如 benchmark_tp.py micro 2 32（bs=32）
"""
import os, sys, time, statistics, json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

MODE = sys.argv[1]
TP = int(sys.argv[2])
N_MULTI = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0=只跑单用户

IN_TOK, OUT_TOK = 256, 768
TEMP = 0.01
ROUNDS = 7
M_IN, M_OUT = 128, 256  # 多用户


def make_prompt(n_tok):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < n_tok:
        ids += tok.encode(" data")
    ids = ids[:n_tok]
    return tok.decode(ids)


def run_micro(tp):
    sys.path.insert(0, REPO_ROOT)
    import torch
    import torch.distributed as dist
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core.parallel_config import get_rank, get_world_size, rank0
    from core import sequence as sm
    # 强制跑满 max_tokens（ignore_eos 语义，对齐 vLLM 口径）
    sm.Sequence.is_finished = lambda s: len(s.output_ids) >= s.max_tokens

    torch.cuda.set_device(int(os.environ.get("RANK", "0")))
    eng = InferenceEngine(MODEL, max_batch_size=max(N_MULTI, 64), max_prefill_tokens=4096)
    rank = get_rank()
    ws = get_world_size()
    assert ws == tp, f"world_size {ws} != 期望 TP {tp}"

    def drive_until_done(eng, rank, ws):
        """rank0 驱动 scheduler + broadcast，非 rank0 receive。跑完当前所有请求。
        ws==1 时不走 broadcast（dist 未初始化），直接单进程循环。"""
        while True:
            if ws == 1:
                b, bt = eng.get_next_batch()
                if not b:
                    break
                ctx = BatchInferenceContext(len(b), bt, b)
                eng.step(ctx)
                eng.collect(ctx)
                eng.update_sequences(ctx.sequences)
                continue
            if rank0():
                b, bt = eng.get_next_batch()
                done = (not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue)
                if bt == "waiting" or not b:
                    BatchInferenceContext(0, "waiting").broadcast()
                else:
                    ctx = BatchInferenceContext(len(b), bt, b)
                    ctx.broadcast()          # bcast1: 完整 seq（建立 batch）
                    eng.step(ctx)
                    eng.collect(ctx)
                    eng.tp_broadcast_tokens(ctx)  # bcast2: 只发 [bs] 采样 token
                    eng.update_sequences(ctx.sequences)
                dt_ = torch.tensor([1 if done else 0], device=eng.device)
                dist.broadcast(dt_, src=0)
            else:
                ctx = BatchInferenceContext.receive(eng.tokenizer)  # bcast1
                if ctx.batch_type != "waiting" and ctx.batch_size > 0:
                    eng.step(ctx)
                    seqs = eng.tp_receive_tokens(ctx)  # bcast2: decode 收 token / prefill 收完整
                    eng.update_sequences(seqs)
                dt_ = torch.zeros(1, device=eng.device)
                dist.broadcast(dt_, src=0)
            if int(dt_.item()) == 1:
                break

    prompt = make_prompt(IN_TOK)
    # warmup（触发 graph 捕获 + sampler 编译）
    if rank0():
        eng.add_request(prompt, 8, temperature=TEMP, top_p=1.0)
    drive_until_done(eng, rank, ws)

    results = {}
    # ---- 单用户 decode：7 轮取中位数 ----
    if rank0():
        walls = []
        for _ in range(ROUNDS):
            eng.add_request(prompt, OUT_TOK, temperature=TEMP, top_p=1.0)
            torch.cuda.synchronize(); t0 = time.time()
            drive_until_done(eng, rank, ws)
            torch.cuda.synchronize(); walls.append(time.time() - t0)
        med = statistics.median(walls)
        results["single_user"] = {
            "tok_s": OUT_TOK / med, "wall_ms": med * 1000,
            "rounds_ms": [round(w * 1000) for w in walls],
        }
        print(f"[micro TP={tp}] single-user ({IN_TOK}in/{OUT_TOK}out): "
              f"{OUT_TOK/med:.1f} tok/s | wall {med*1000:.0f}ms | "
              f"rounds={[f'{w*1000:.0f}' for w in walls]}", flush=True)

    # ---- 多用户吞吐 ----
    if N_MULTI > 0:
        mprompt = make_prompt(M_IN)
        if rank0():
            for _ in range(N_MULTI):
                eng.add_request(mprompt, M_OUT, temperature=TEMP, top_p=1.0)
            torch.cuda.synchronize(); t0 = time.time()
        drive_until_done(eng, rank, ws)
        if rank0():
            torch.cuda.synchronize(); dt = time.time() - t0
            results["multi_user"] = {"bs": N_MULTI, "tok_s": N_MULTI * M_OUT / dt,
                                     "wall_s": dt}
            print(f"[micro TP={tp}] multi-user bs={N_MULTI} ({M_IN}in/{M_OUT}out): "
                  f"{N_MULTI*M_OUT/dt:.1f} tok/s ({N_MULTI*M_OUT} tok, {dt:.3f}s)", flush=True)

    if rank0():
        mem_mb = torch.cuda.memory_allocated() / (1 << 20)
        results["mem_mb"] = mem_mb
        print(f"[micro TP={tp}] rank0 mem={mem_mb:.0f}MB", flush=True)
        with open(f"/tmp/bench_tp_micro_{tp}.json", "w") as f:
            json.dump(results, f)


def run_vllm(tp):
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=tp, max_model_len=4096)
    prompt = make_prompt(IN_TOK)
    llm.generate([prompt], SamplingParams(temperature=TEMP, max_tokens=8))
    walls = []
    for _ in range(ROUNDS):
        t0 = time.time()
        llm.generate([prompt], SamplingParams(temperature=TEMP, ignore_eos=True,
                                              max_tokens=OUT_TOK), use_tqdm=False)
        walls.append(time.time() - t0)
    med = statistics.median(walls)
    print(f"[vllm TP={tp}] single-user ({IN_TOK}in/{OUT_TOK}out): "
          f"{OUT_TOK/med:.1f} tok/s | wall {med*1000:.0f}ms | "
          f"rounds={[f'{w*1000:.0f}' for w in walls]}", flush=True)
    results = {"single_user": {"tok_s": OUT_TOK / med, "wall_ms": med * 1000,
                               "rounds_ms": [round(w * 1000) for w in walls]}}
    if N_MULTI > 0:
        mprompt = make_prompt(M_IN)
        prompts = [mprompt] * N_MULTI
        sps = [SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=M_OUT)
               for _ in range(N_MULTI)]
        t0 = time.time()
        llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - t0
        results["multi_user"] = {"bs": N_MULTI, "tok_s": N_MULTI * M_OUT / dt, "wall_s": dt}
        print(f"[vllm TP={tp}] multi-user bs={N_MULTI} ({M_IN}in/{M_OUT}out): "
              f"{N_MULTI*M_OUT/dt:.1f} tok/s ({N_MULTI*M_OUT} tok, {dt:.3f}s)", flush=True)
    with open(f"/tmp/bench_tp_vllm_{tp}.json", "w") as f:
        json.dump(results, f)


if MODE == "micro":
    run_micro(TP)
elif MODE == "vllm":
    run_vllm(TP)
else:
    raise SystemExit(f"unknown mode: {MODE} (micro|vllm)")
