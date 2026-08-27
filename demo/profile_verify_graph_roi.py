"""CUDA Graph ROI 判定：spec verify / draft 的 launch gap 实测。

任务背景：假设 verify 52ms 里约 20ms 是 launch gap + 细碎算子低效，CUDA graph 可
砍到 32-38ms。本脚本用两个独立信号判定该假设是否成立：

  1. torch.profiler 的 Self CUDA time（纯 GPU kernel 时间，不含 CPU/launch 间隙）。
     若一个 verify 的 wall≈52ms 而 Self CUDA 也≈52ms，则 GPU 几乎一直在算，
     launch gap≈0，CUDA graph 无用武之地。
  2. 紧循环 wall 计时（sync 包夹）vs kernel 时间 → gap = wall - kernel。

结论（本分支 main 实测，见 commit）：verify 是 GPU memory-bound（int8 GEMM 权重读
主导，~67% HBM），launch gap 仅 ~0.2-0.5ms/step（0.3%）。CUDA graph 对 spec 路径
的 ROI ≈ 0.3%，非 15ms。与历史结论一致（CUDA graph 已吃掉 launch 间隙，瓶颈在
HBM 权重读，非 dispatch）。

用法：CUDA_VISIBLE_DEVICES=4 MICRO_W8A16=1 python3 demo/profile_verify_graph_roi.py
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from torch.profiler import profile, ProfilerActivity
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
N_ITER = int(os.environ.get("N_ITER", "20"))


def _cuda_total_ms(prof):
    """profile 的 Self CUDA time 总和（ms）。"""
    return sum(e.self_device_time_total for e in prof.key_averages()) / 1e3


@torch.inference_mode()
def main():
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096,
                          spec_decode=True, draft_model_path=DRAFT,
                          num_speculative_tokens=7)
    ctrl = eng._spec_engine
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

    # 构造一个 verify 闭包（非首步，走 init_from_cp，代表稳态）
    def make_verify():
        nonlocal anchor, kv_len, accepted_prev
        draft_tokens = ctrl._draft_propose(anchor, kv_len)
        vid = torch.cat([torch.tensor([anchor], device=device), draft_tokens])
        vpos = torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64)
        vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
        vcu_q = torch.tensor([0, M], device=device, dtype=torch.int32)
        vcu_k = torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32)
        init_kw = {}
        if accepted_prev is not None:
            init_kw = dict(init_from_cp=True,
                           init_state_s=ctrl._gdn_cp_state[accepted_prev],
                           init_state_c=ctrl._gdn_cp_conv[accepted_prev])
        def run():
            nonlocal anchor, kv_len, accepted_prev
            vl = ctrl._target_forward(vid, vpos, vslot, vcu_q, vcu_k, bt, M,
                                      kv_len - 1 + M, gdn_slot,
                                      collect_aux_from=kv_len - 1, gdn_checkpoint=True, **init_kw)
            t_cpu = vl.argmax(dim=-1).cpu().tolist()
            d_cpu = draft_tokens.cpu().tolist()
            acc = 0
            for i in range(N):
                if d_cpu[i] == t_cpu[i]:
                    acc += 1
                else:
                    break
            accepted_prev = acc
            anchor = t_cpu[acc]
            kv_len += acc + 1
        return run

    # warm（触发编译 + 稳定）
    for _ in range(3):
        make_verify()()

    # ---------- verify：wall vs kernel ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        make_verify()()
    torch.cuda.synchronize()
    verify_wall = (time.perf_counter() - t0) / N_ITER * 1000

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            make_verify()()
        torch.cuda.synchronize()
    verify_kernel = _cuda_total_ms(prof) / 5
    verify_gap = verify_wall - verify_kernel

    # verify kernel top（定位细碎算子 vs 大 GEMM）
    print(f"\n=== verify（M={M}，稳态，warm）===")
    print(f"  wall（紧循环，含 draft+accept 的 .cpu 同步）: {verify_wall:.2f} ms")
    print(f"  kernel（Self CUDA，纯 GPU）:                 {verify_kernel:.2f} ms")
    print(f"  launch gap = wall - kernel:                  {verify_gap:.2f} ms  "
          f"({verify_gap/verify_wall*100:.1f}%)")
    print("  top kernels:")
    for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:12]:
        if e.self_device_time_total > 0:
            print(f"    {e.key[:52]:52s} {e.self_device_time_total/5/1e3:8.3f} ms "
                  f"({e.count//5:4d}x/verify, avg {e.self_device_time_total/5/e.count*1e3:8.2f} us)")

    # ---------- draft：wall vs kernel ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        ctrl._draft_propose(anchor, kv_len)
    torch.cuda.synchronize()
    draft_wall = (time.perf_counter() - t0) / N_ITER * 1000
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as profd:
        for _ in range(5):
            ctrl._draft_propose(anchor, kv_len)
        torch.cuda.synchronize()
    draft_kernel = _cuda_total_ms(profd) / 5
    draft_gap = draft_wall - draft_kernel
    print(f"\n=== draft（N={N}）===")
    print(f"  wall（紧循环，含 .cpu 同步）:  {draft_wall:.2f} ms")
    print(f"  kernel（Self CUDA）:          {draft_kernel:.2f} ms")
    print(f"  launch gap = wall - kernel:   {draft_gap:.2f} ms  ({draft_gap/draft_wall*100:.1f}%)")

    # ---------- 全 step（draft+verify+accept）gap，交叉验证 ----------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        make_verify()()  # 内含 draft_propose + verify + accept(.cpu)
    torch.cuda.synchronize()
    step_wall = (time.perf_counter() - t0) / N_ITER * 1000
    print(f"\n=== 全 step（draft+verify+accept）===")
    print(f"  wall: {step_wall:.2f} ms")
    print(f"  若 CUDA graph 只省 launch gap：省 ~{verify_gap + draft_gap:.2f} ms "
          f"= {(verify_gap + draft_gap)/step_wall*100:.1f}% of step")

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
