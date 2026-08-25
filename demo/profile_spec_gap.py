"""spec decode 每步 e2e 缺口定位：真实 step wall time 分解。

每步（draft+verify+accept）：
  - wall: 步首 sync 后 perf_counter 到步尾 sync 后（真实 step wall time）
  - cpu_*: 各段纯 CPU 时间（perf_counter 包夹，无 sync，= Python 执行 + launch 排队）
  - gpu: CUDA event 量步内 GPU 忙时间（event 在 stream 上，含 kernel 间 gap）
  - 缺口 = wall - gpu_busy（CPU 侧 launch 间隙 + 同步等待 + Python）

用法：CUDA_VISIBLE_DEVICES=4 MICRO_W8A16=1 python3 demo/profile_spec_gap.py
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
N_ITER = int(os.environ.get("N_ITER", "10"))


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

    # ---- 与 generate() 主循环一致的单步函数（无 per-segment sync）----
    def one_step():
        t0 = time.perf_counter()
        draft_tokens = ctrl._draft_propose(anchor, kv_len)
        t1 = time.perf_counter()
        ctrl._verify_ids[0] = anchor
        ctrl._verify_ids[1:] = draft_tokens
        vpos = ctrl._arange[kv_len - 1: kv_len - 1 + M]
        vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
        ctrl._cu_q[0] = 0
        ctrl._cu_q[1] = M
        ctrl._cu_k[0] = 0
        ctrl._cu_k[1] = kv_len - 1 + M
        if ctrl._gdn_accepted_prev is not None:
            vlogits = ctrl._target_forward(
                ctrl._verify_ids, vpos, vslot, ctrl._cu_q, ctrl._cu_k, bt,
                M, kv_len - 1 + M, gdn_slot,
                collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                init_from_cp=True,
                init_state_s=ctrl._gdn_cp_state[ctrl._gdn_accepted_prev],
                init_state_c=ctrl._gdn_cp_conv[ctrl._gdn_accepted_prev])
        else:
            vlogits = ctrl._target_forward(
                ctrl._verify_ids, vpos, vslot, ctrl._cu_q, ctrl._cu_k, bt,
                M, kv_len - 1 + M, gdn_slot,
                collect_aux_from=kv_len - 1, gdn_checkpoint=True)
        t2 = time.perf_counter()
        target_preds = vlogits.argmax(dim=-1)
        d_cpu = draft_tokens.cpu().tolist()
        t_cpu = target_preds.cpu().tolist()
        accepted = 0
        for i in range(N):
            if d_cpu[i] == t_cpu[i]:
                accepted += 1
            else:
                break
        bonus = t_cpu[accepted]
        t3 = time.perf_counter()
        ctrl._gdn_accepted_prev = accepted
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, accepted, bonus

    # warm（编译 + 稳定）
    for _ in range(3):
        _, _, _, accepted, bonus = one_step()
        anchor = bonus
        kv_len += accepted + 1

    # ---- 计时：wall（sync 包夹）+ cpu 分段 + gpu event ----
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    walls, dcs, vcs, acs, gpus = [], [], [], [], []
    for it in range(N_ITER):
        torch.cuda.synchronize()
        ev0.record()
        tw0 = time.perf_counter()
        dc, vc, ac, accepted, bonus = one_step()
        ev1.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - tw0) * 1000)
        dcs.append(dc); vcs.append(vc); acs.append(ac)
        gpus.append(ev0.elapsed_time(ev1))
        anchor = bonus
        kv_len += accepted + 1

    n = len(walls)
    print(f"\n=== spec step 分解（ms/step，均值 over {n} iters）===")
    print(f"  {'wall (sync 包夹)':24s} {sum(walls)/n:8.3f}")
    print(f"  {'  gpu busy (event)':24s} {sum(gpus)/n:8.3f}")
    print(f"  {'  缺口 wall-gpu':24s} {sum(walls)/n - sum(gpus)/n:8.3f}")
    print(f"  {'cpu draft (无sync)':24s} {sum(dcs)/n:8.3f}")
    print(f"  {'cpu verify (无sync)':24s} {sum(vcs)/n:8.3f}")
    print(f"  {'cpu accept (含2次.cpu)':24s} {sum(acs)/n:8.3f}")
    print(f"  {'cpu 三段和':24s} {sum(dcs)/n + sum(vcs)/n + sum(acs)/n:8.3f}")
    print(f"  wall 明细: {[f'{w:.1f}' for w in walls]}")
    print(f"  gpu  明细: {[f'{g:.1f}' for g in gpus]}")

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
