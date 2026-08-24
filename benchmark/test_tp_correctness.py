"""TP 正确性验证：TP=1 单卡 vs TP=2 (torchrun) 逐 token 对比。

用法：
  TP=1:  CUDA_VISIBLE_DEVICES=2 TP_OUT_DIR=/tmp/tp1_out python3 benchmark/test_tp_correctness.py
  TP=2:  CUDA_VISIBLE_DEVICES=2,5 NCCL_P2P_DISABLE=1 TP_OUT_DIR=/tmp/tp2_out \
          torchrun --nproc_per_node=2 --master_port=29513 benchmark/test_tp_correctness.py

TP 协议（对齐 api_server.py）：仅 rank0 驱动 scheduler 并 broadcast，非 rank0 receive。
  rank0:     get_next_batch → ctx.broadcast() → step → collect → ctx.broadcast() → update_sequences
  non-rank0: ctx=receive() → step → ctx=receive() → update_sequences
  退出：rank0 用 dist 广播 done 标志。

关键设计（避免 teardown 死锁）：
  - rank/ws 在 InferenceEngine 构造【之后】取（setup() 在 __init__ 里才初始化 dist，
    之前 get_rank()/get_world_size() 恒返回 0/1 → 两 rank 误走单进程路径 → 死锁）。
  - 循环结束后各 rank【立即】写自己的 JSON，不做任何 post-loop collective
    （barrier/all_gather 会让先完成的 rank0 等 rank1，而 rank1 还在收尾 broadcast → 死锁）。
    数据落盘后再退出，teardown 是否干净都不影响结果。
  - 外部脚本对比 /tmp/tp1_out/tp2_rank0.json 与 /tmp/tp2_out/tp2_rank0.json 的 tokens。
"""
import os, sys, json, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")
N_TOKENS = 200
OUT_DIR = os.environ.get("TP_OUT_DIR", "/tmp")

PROMPT = ("The history of artificial intelligence began in the mid 20th century. "
          "It has been a field of study since the 1950s, and has seen many "
          "breakthroughs in machine learning, neural networks, and deep learning. "
          "Please continue this paragraph about the development of AI research.")


def main():
    import torch
    import torch.distributed as dist
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core.parallel_config import get_rank, get_world_size, rank0

    # 预置设备（engine 内部 _init_distributed 会再 set_device(self.rank)）。
    torch.cuda.set_device(int(os.environ.get("RANK", "0")))

    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    # rank/ws 必须在构造【之后】取（setup() 在 __init__ 里才初始化 dist）。
    rank = get_rank()
    ws = get_world_size()
    torch.cuda.synchronize()
    mem_mb = torch.cuda.memory_allocated() / (1 << 20)

    diag = {
        "rank": rank, "world_size": ws,
        "dist_initialized": dist.is_initialized(),
        "env_RANK": os.environ.get("RANK"), "env_WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "num_heads": eng.num_heads, "kv_num_heads": eng.kv_num_heads,
        "intermediate": eng.intermediate_size, "mem_mb": mem_mb,
    }
    print(f"[diag] {json.dumps(diag)}", flush=True)

    if rank0():
        eng.add_request(PROMPT, N_TOKENS, temperature=0.0, top_p=1.0)

    margins = []
    last_seqs = None  # 非 rank0：最后一次收到的 seq 状态（含全部 output_ids）
    # 终止协议：rank0 的 get_next_batch 返回空 batch（idle）时广播一个 "waiting" ctx 作为
    # 结束信号，两 rank 都收到 waiting 即 break。不用额外的 done_t broadcast（那是独立
    # collective，收尾时 rank0 先 break 退出 main→atexit 销毁进程组，rank1 还在 done_t 上
    # → 死锁）。waiting ctx 复用已有的 ctx broadcast 通道，无额外 collective。
    if ws > 1:
        while True:
            if rank0():
                batch, batch_type = eng.get_next_batch()
                if not batch:  # idle：无工作 → 广播结束信号
                    BatchInferenceContext(0, "waiting").broadcast()
                    break
                ctx = BatchInferenceContext(len(batch), batch_type, batch)
                ctx.broadcast()
                eng.step(ctx)
                if batch_type == "decode":
                    top2 = torch.topk(ctx.logits[0], 2, dim=-1).values
                    margins.append(float(top2[0] - top2[1]))
                eng.collect(ctx)
                ctx.broadcast()
                eng.update_sequences(ctx.sequences)
            else:
                ctx = BatchInferenceContext.receive(eng.tokenizer)
                if ctx.batch_type == "waiting" or ctx.batch_size == 0:
                    break
                eng.step(ctx)
                ctx = BatchInferenceContext.receive(eng.tokenizer)
                eng.update_sequences(ctx.sequences)
                last_seqs = ctx.sequences
    else:
        while True:
            b, bt = eng.get_next_batch()
            if not b:
                break
            ctx = BatchInferenceContext(len(b), bt, b)
            eng.step(ctx)
            if bt == "decode":
                top2 = torch.topk(ctx.logits[0], 2, dim=-1).values
                margins.append(float(top2[0] - top2[1]))
            eng.collect(ctx)
            eng.update_sequences(ctx.sequences)

    # 不做 post-loop collective（barrier 会让先 break 的 rank0 等 rank1，而 rank1 可能
    # 还在收尾 → 死锁）。各 rank 直接写 JSON 退出，数据落盘后再 teardown。
    # 收集本 rank 的 tokens：rank0 从 scheduler；非 rank0 从最后收到的 seq 状态。
    tokens = []
    if rank0():
        if eng.scheduler.finished_sequences:
            tokens = eng.scheduler.finished_sequences[0].output_ids
    elif last_seqs:
        seen = {}
        for s in last_seqs:
            if s.seq_id not in seen or len(s.output_ids) > len(seen[s.seq_id]):
                seen[s.seq_id] = s.output_ids
        if seen:
            tokens = max(seen.values(), key=len)

    # 立即写本 rank 的 JSON（不做 post-loop collective，避免 teardown 死锁）。
    text = eng.tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
    print(f"[TP={ws}] rank{rank}: {len(tokens)} tokens, mem={mem_mb:.0f}MB", flush=True)
    if rank0():
        print(f"[TP={ws}] text: {text[:300]}", flush=True)
        if margins:
            print(f"[TP={ws}] margins min={min(margins):.4f} mean={sum(margins)/len(margins):.4f}",
                  flush=True)
    with open(os.path.join(OUT_DIR, f"tp2_rank{rank}.json"), "w") as f:
        json.dump({"rank": rank, "world_size": ws, "tokens": tokens,
                   "margins": margins if rank0() else None, "mem_mb": mem_mb,
                   "diag": diag}, f)
    print(f"[TP={ws}] rank{rank}: wrote {os.path.join(OUT_DIR, f'tp2_rank{rank}.json')}", flush=True)


if __name__ == "__main__":
    main()
