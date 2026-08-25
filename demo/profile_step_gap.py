"""130ms 缺口定位：spec decode 每步「e2e 隐含 step 时间」vs「segment 和」的差。

方法（coordinator 指定）：
  1. 全 step 同步计时：step 起止 cuda.synchronize 包夹，量 e2e 隐含 step 时间。
  2. segment 和：draft(6 段) + verify + accept 各段单独 sync 计时之和。
  3. 缺口 = e2e step 时间 - segment 和（CPU Python + launch gap + sync 开销）。
  4. cProfile generate 循环 → CPU Python 时间分解。

注意：specopt 分支 verify GEMM 是旧 CUDA tiled GEMV（~500ms），GPU 忙 500ms，
CPU 有充足时间 launch 下一段 → 缺口小。dflash 分支（TileLang GEMM，38.3 tok/s）
verify GEMM 快得多，GPU 早完成，CPU 开销才显形为大缺口（~130ms）。本脚本在
specopt 上量「segment 和 vs e2e step」的差，作为参照系。

【结论（两分支实测，N_ITER=5）】
  specopt（旧 CUDA GEMV，verify 500ms）：segment 和 510.970ms，e2e step 511.152ms，
    缺口 0.182ms。GPU 忙 500ms，CPU 有充足时间 launch → 缺口极小。
  dflash（TileLang GEMM，verify 52.5ms）：segment 和 65.689ms，e2e step 65.969ms，
    缺口 0.280ms。decode 单步仅 66ms，【无 130ms 缺口】。
  cProfile（两分支）：CPU Python 可忽略（accept 的 .cpu() 是 GPU-wait 非 CPU 计算；
    triton/tilelang launch 每 step 仅 ~10-20ms 且被 GPU 时间掩盖）。

  「130ms 缺口」真相：dflash e2e 38.3 tok/s = 64 tok / 1.67s = prefill(~1.08s,
  61 tok) + 9 decode step(9×66ms=0.594s)。1.08s prefill / 9 step ≈ 120ms/step =
  所谓「130ms 缺口」实为【一次性 prompt prefill 摊到 decode step】，非每步
  CPU/Python/launch 开销（实测仅 0.28ms）。e2e tok/s 指标把 prefill 算进分子分母，
  拉低了表观吞吐；纯 decode 稳态是 66ms/step ≈ 150 tok/s（acceptance 6.556）。
  要提 e2e tok/s 应优化 prefill（或加长输出摊薄 prefill），而非抠 decode 每步开销。

用法：CUDA_VISIBLE_DEVICES=3 MICRO_W8A16=1 N_ITER=5 python3 demo/profile_step_gap.py
"""
import os, sys, time, cProfile, pstats, io
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
N_ITER = int(os.environ.get("N_ITER", "5"))


def _accept(draft_tokens, vlogits, N):
    target_preds = vlogits.argmax(dim=-1)
    d_cpu = draft_tokens.cpu().tolist()
    t_cpu = target_preds.cpu().tolist()
    accepted = 0
    for i in range(N):
        if d_cpu[i] == t_cpu[i]:
            accepted += 1
        else:
            break
    return accepted, t_cpu[accepted]


@torch.inference_mode()
def main():
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096,
                          spec_decode=True, draft_model_path=DRAFT,
                          num_speculative_tokens=7)
    ctrl = eng._spec_controller
    device = eng.device
    N = ctrl.N
    M = 1 + N
    prompt_ids = torch.tensor(eng.tokenizer.encode(PROMPT, add_special_tokens=True),
                              dtype=torch.int64, device=device)
    P = prompt_ids.shape[0]

    seq_id = 1_000_000
    ok, slot_mapping, _ = eng.cache_manager.alloc(seq_id, ctrl.max_len, None)
    blocks = eng.cache_manager._blocks[seq_id]
    bt = torch.full((1, eng.cache_manager.max_seq_blocks), -1, dtype=torch.int32, device=device)
    bt[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
    gdn_slot = ctrl._gdn_alloc()
    eng.prefill_runner._gdn_state_pool[gdn_slot] = 0
    eng.prefill_runner._gdn_conv_state_pool[gdn_slot] = 0

    positions = torch.arange(P, device=device, dtype=torch.int64)
    cu_q = torch.tensor([0, P], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, P], device=device, dtype=torch.int32)
    logits = ctrl._target_forward(prompt_ids, positions, slot_mapping[:P], cu_q, cu_k, bt,
                                  P, P, gdn_slot, collect_aux_from=0, gdn_checkpoint=False)
    anchor = int(logits[-1].argmax())
    kv_len = P + 1
    accepted_prev = None

    # ---- 一个完整 step（draft + verify + accept），返回 (anchor, kv_len, seg_dict) ----
    def run_step(timing=False):
        nonlocal anchor, kv_len, accepted_prev
        ctx_len = kv_len - 1
        seg = {}

        def timeit(fn):
            if timing:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fn()
            if timing:
                torch.cuda.synchronize()
                seg[_cur[0]] = (time.perf_counter() - t0) * 1000
            return r

        _cur[0] = "draft.aux_gather"
        aux = timeit(lambda: ctrl.aux_cache[:, :ctx_len].permute(1, 0, 2).reshape(ctx_len, -1))
        _cur[0] = "draft.combine_fc"
        combined = timeit(lambda: ctrl.draft.combine_hidden_states(aux))
        _cur[0] = "draft.precompute_ctx_kv"
        context_kv = timeit(lambda: ctrl.draft.precompute_context_kv(combined, ctrl._arange[:ctx_len]))
        ctrl._query_ids[0] = anchor
        _cur[0] = "draft.query_embed"
        query_embeds = timeit(lambda: ctrl.embed(ctrl._query_ids))
        query_pos = ctrl._arange[kv_len - 1: kv_len - 1 + 1 + N]
        _cur[0] = "draft.forward_5L"
        out = timeit(lambda: ctrl.draft.forward(ctrl._query_ids, query_pos,
                                                input_embeds=query_embeds, context_kv=context_kv))
        ctrl._anchor_t[0] = anchor
        _cur[0] = "draft.select_tokens"
        draft = timeit(lambda: ctrl.draft.select_draft_tokens(out[1:].unsqueeze(0), ctrl._anchor_t))
        draft_tokens = draft[0]

        if accepted_prev is not None:
            _cur[0] = "verify.target_forward"
            vlogits = timeit(lambda: ctrl._target_forward(
                torch.cat([torch.tensor([anchor], device=device), draft_tokens]),
                torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64),
                slot_mapping[kv_len - 1: kv_len - 1 + M],
                torch.tensor([0, M], device=device, dtype=torch.int32),
                torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32),
                bt, M, kv_len - 1 + M, gdn_slot,
                collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                init_from_cp=True,
                init_state_s=ctrl._gdn_cp_state[accepted_prev],
                init_state_c=ctrl._gdn_cp_conv[accepted_prev]))
        else:
            _cur[0] = "verify.target_forward"
            vlogits = timeit(lambda: ctrl._target_forward(
                torch.cat([torch.tensor([anchor], device=device), draft_tokens]),
                torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64),
                slot_mapping[kv_len - 1: kv_len - 1 + M],
                torch.tensor([0, M], device=device, dtype=torch.int32),
                torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32),
                bt, M, kv_len - 1 + M, gdn_slot,
                collect_aux_from=kv_len - 1, gdn_checkpoint=True))

        _cur[0] = "accept"
        res = timeit(lambda: _accept(draft_tokens, vlogits, N))
        accepted, bonus = res
        accepted_prev = accepted
        anchor = bonus
        kv_len += accepted + 1
        return seg

    _cur = [""]

    # warm
    for _ in range(2):
        run_step(timing=False)

    # ---- Phase 1: 全 step 同步计时 + segment 和 ----
    step_times = []
    seg_sum = {}
    for it in range(N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seg = run_step(timing=True)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        for k, v in seg.items():
            seg_sum.setdefault(k, []).append(v)

    avg_step = sum(step_times) / len(step_times)
    seg_total = {k: sum(v) / len(v) for k, v in seg_sum.items()}
    seg_sum_total = sum(seg_total.values())
    gap = avg_step - seg_sum_total

    print("\n=== 全 step 同步计时（ms/step，均值 over %d iters）===" % N_ITER)
    for k in ["draft.aux_gather", "draft.combine_fc", "draft.precompute_ctx_kv",
              "draft.query_embed", "draft.forward_5L", "draft.select_tokens",
              "verify.target_forward", "accept"]:
        print(f"  {k:26s} {seg_total[k]:8.3f} ms")
    print(f"  {'segment 和':26s} {seg_sum_total:8.3f} ms")
    print(f"  {'e2e step（sync 包夹）':26s} {avg_step:8.3f} ms   (min {min(step_times):.3f} max {max(step_times):.3f})")
    print(f"  {'缺口 = e2e - segment和':26s} {gap:8.3f} ms")

    # ---- Phase 2: cProfile generate 循环（CPU Python 分解）----
    print("\n=== cProfile（CPU Python 时间，%d steps）===" % N_ITER)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N_ITER):
        run_step(timing=False)
    pr.disable()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(18)
    print(s.getvalue())
    print(f"  cProfile wall（含 profiler 开销）: {wall:.1f} ms / {N_ITER} steps = {wall/N_ITER:.3f} ms/step")
    print(f"  cProfile total CPU（tottime 和）: {sum(v[2] for v in pr.stats.values())*1000/N_ITER:.3f} ms/step")

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
