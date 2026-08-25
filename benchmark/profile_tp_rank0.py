"""TP rank0 驱动循环 Python 热点剖析（cProfile + 逐段细计时）。

目的：定位 micro TP=2 每步 4.05ms 里 Python 侧（非 GPU forward）到底花在哪。
coordinator 怀疑 update_sequences 对完整 token 历史做 O(L) 操作。

用法（rank0 驱动，非 rank0 同步）：
  CUDA_VISIBLE_DEVICES=2,5 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 \
    --master_port=29730 benchmark/profile_tp_rank0.py 32
"""
import os, sys, time, cProfile, pstats, io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
M_IN, M_OUT, M_TEMP = 128, 256, 0.01
N_PROFILE = int(os.environ.get("N_PROFILE", "200"))  # cProfile 覆盖的 decode 步数


def make_prompt(n_tok):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < n_tok:
        ids += tok.encode(" data")
    return tok.decode(ids[:n_tok])


def main():
    import torch
    import torch.distributed as dist
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core.parallel_config import get_rank, rank0

    rank = get_rank()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    eng = InferenceEngine(MODEL, max_batch_size=max(BS, 64), max_prefill_tokens=4096)
    prompt = make_prompt(M_IN)

    # warmup（graph 捕获 + 算法选择）
    if rank0():
        eng.add_request(prompt, 8, temperature=M_TEMP, top_p=1.0)
    _loop(eng, max_steps=8)

    if rank0():
        for _ in range(BS):
            eng.add_request(prompt, M_OUT, temperature=M_TEMP, top_p=1.0)
    torch.cuda.synchronize()

    # ---- 阶段 A：细计时（无 cProfile 开销），拆 rank0 每步各子操作 ----
    # 两 rank 都跑（保持 NCCL 同步）；rank0 计时，非 rank0 镜像。
    t_acc = {}
    steps = 0
    while steps < N_PROFILE:
        if rank0():
            b, bt = eng.get_next_batch()
            done = (not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue)
            if bt == "waiting" or not b:
                eng.tp_broadcast_waiting(done)
                if done:
                    break
                steps += 1
                continue
            t0 = time.perf_counter()
            ctx = BatchInferenceContext(len(b), bt, b)
            t1 = time.perf_counter()
            eng.tp_broadcast_batch(ctx, done)
            t2 = time.perf_counter()
            eng.step(ctx)
            t3 = time.perf_counter()
            eng.collect(ctx)
            t4 = time.perf_counter()
            eng.tp_broadcast_tokens(ctx)
            t5 = time.perf_counter()
            eng.update_sequences(ctx.sequences)
            t6 = time.perf_counter()
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            t_acc["ctx_ctor"] = t_acc.get("ctx_ctor", 0) + (t1 - t0)
            t_acc["bcast1"] = t_acc.get("bcast1", 0) + (t2 - t1)
            t_acc["step_submit"] = t_acc.get("step_submit", 0) + (t3 - t2)
            t_acc["collect"] = t_acc.get("collect", 0) + (t4 - t3)
            t_acc["bcast2"] = t_acc.get("bcast2", 0) + (t5 - t4)
            t_acc["upd_seq"] = t_acc.get("upd_seq", 0) + (t6 - t5)
            t_acc["wall_sync"] = t_acc.get("wall_sync", 0) + (t7 - t0)
            if done:
                break
        else:
            ctx, done = eng.tp_receive_batch()
            if ctx.batch_type != "waiting" and ctx.batch_size > 0:
                eng.step(ctx)
                seqs = eng.tp_receive_tokens(ctx)
                eng.update_sequences(seqs)
            if done:
                break
        steps += 1
    if rank0():
        n = max(steps, 1)
        print(f"[FINE rank0] steps={n} per-step(ms):", flush=True)
        for k in ["wall_sync", "step_submit", "collect", "bcast1", "bcast2", "upd_seq", "ctx_ctor"]:
            print(f"    {k:12} = {t_acc.get(k,0)*1000/n:.3f}", flush=True)

    # ---- 阶段 B：cProfile 覆盖 N_PROFILE 步，看 Python 函数累计时间分布 ----
    # 重新加请求（上一轮已跑完）；两 rank 都跑保持同步，rank0 出 profile。
    if rank0():
        for _ in range(BS):
            eng.add_request(prompt, M_OUT, temperature=M_TEMP, top_p=1.0)
    torch.cuda.synchronize()
    prof = cProfile.Profile()
    prof.enable()
    _loop(eng, max_steps=N_PROFILE)
    prof.disable()
    torch.cuda.synchronize()
    if rank0():
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
        ps.print_stats(40)
        print("[CPROFILE rank0] top by cumulative:", flush=True)
        print(s.getvalue(), flush=True)


def _loop(eng, max_steps):
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
                    eng.tp_broadcast_waiting(done)
                else:
                    ctx = BatchInferenceContext(len(b), bt, b)
                    eng.tp_broadcast_batch(ctx, done)
                    eng.step(ctx)
                    eng.collect(ctx)
                    eng.tp_broadcast_tokens(ctx)
                    eng.update_sequences(ctx.sequences)
                if done:
                    break
            else:
                ctx, done = eng.tp_receive_batch()
                if ctx.batch_type != "waiting" and ctx.batch_size > 0:
                    eng.step(ctx)
                    seqs = eng.tp_receive_tokens(ctx)
                    eng.update_sequences(seqs)
                if done:
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


if __name__ == "__main__":
    main()
