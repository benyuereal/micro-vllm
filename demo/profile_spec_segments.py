"""细粒度分段计时（差分法）：spec decode 每步热路径各段非 GEMM 耗时。

段划分：
  draft: aux_gather / combine(fc) / precompute_ctx_kv / query_embed /
         forward_5L / select_tokens(lm_head+selector)
  verify: _forward 整体（GEMM 主导，另列非 GEMM 参考）
  accept: argmax + 贪心比较 + bonus（2 次 CPU-GPU 同步）
  rollback: 去 rollback 后为 0（GDN 初始状态直接读 checkpoint[accepted_prev]，
            省掉原 0.406ms/step 的 DtoD copy_ 回 pool）

用法：CUDA_VISIBLE_DEVICES=3 MICRO_W8A16=1 python3 demo/profile_spec_segments.py
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
N_ITER = int(os.environ.get("N_ITER", "5"))


def _accept(draft_tokens, vlogits, N):
    # 与 generate() 一致：draft/target 各 .cpu() 一次（共 2 次同步），Python int 比较。
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
    return accepted, bonus


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
    logits = ctrl._forward(prompt_ids, positions, slot_mapping[:P], cu_q, cu_k, bt,
                                  P, P, gdn_slot, collect_aux_from=0, gdn_checkpoint=False)
    anchor = int(logits[-1].argmax())
    kv_len = P + 1

    # warm（编译 + 稳定）。accepted_prev 跟踪上一步接受数（去 rollback：下一步 verify
    # 的 GDN 初始状态直接读 checkpoint[accepted_prev]，与 generate() 一致）。
    accepted_prev = None
    for _ in range(2):
        dt = ctrl._draft_propose(anchor, kv_len)
        vid = torch.cat([torch.tensor([anchor], device=device), dt])
        vpos = torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64)
        vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
        vcu_q = torch.tensor([0, M], device=device, dtype=torch.int32)
        vcu_k = torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32)
        if accepted_prev is not None:
            vl = ctrl._forward(vid, vpos, vslot, vcu_q, vcu_k, bt, M, kv_len - 1 + M,
                                      gdn_slot, collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                                      init_from_cp=True,
                                      init_state_s=ctrl._gdn_cp_state[accepted_prev],
                                      init_state_c=ctrl._gdn_cp_conv[accepted_prev])
        else:
            vl = ctrl._forward(vid, vpos, vslot, vcu_q, vcu_k, bt, M, kv_len - 1 + M,
                                      gdn_slot, collect_aux_from=kv_len - 1, gdn_checkpoint=True)
        accepted_prev = 0
        anchor = int(vl[0].argmax()); kv_len += 1

    # ---- 分段计时（每段单独 sync，串行累加）----
    seg2 = {}
    for it in range(N_ITER):
        ctx_len = kv_len - 1

        def timeit(fn):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000, r

        ms, aux = timeit(lambda: ctrl.aux_cache[:, :ctx_len].permute(1, 0, 2).reshape(ctx_len, -1))
        seg2.setdefault("draft.aux_gather", []).append(ms)
        ms, combined = timeit(lambda: ctrl.draft.combine_hidden_states(aux))
        seg2.setdefault("draft.combine_fc", []).append(ms)
        ms, context_kv = timeit(lambda: ctrl.draft.precompute_context_kv(combined, ctrl._arange[:ctx_len]))
        seg2.setdefault("draft.precompute_ctx_kv", []).append(ms)
        ctrl._query_ids[0] = anchor
        ms, query_embeds = timeit(lambda: ctrl.embed(ctrl._query_ids))
        seg2.setdefault("draft.query_embed", []).append(ms)
        query_pos = ctrl._arange[kv_len - 1: kv_len - 1 + 1 + N]
        ms, out = timeit(lambda: ctrl.draft.forward(ctrl._query_ids, query_pos,
                                                    input_embeds=query_embeds, context_kv=context_kv))
        seg2.setdefault("draft.forward_5L", []).append(ms)
        ctrl._anchor_t[0] = anchor
        ms, draft = timeit(lambda: ctrl.draft.select_draft_tokens(out[1:].unsqueeze(0), ctrl._anchor_t))
        seg2.setdefault("draft.select_tokens", []).append(ms)
        draft_tokens = draft[0]

        # 去 rollback：verify 的 GDN 初始状态直接读 checkpoint[accepted_prev]（非首步）。
        if accepted_prev is not None:
            ms, vlogits = timeit(lambda: ctrl._forward(
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
            ms, vlogits = timeit(lambda: ctrl._forward(
                torch.cat([torch.tensor([anchor], device=device), draft_tokens]),
                torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64),
                slot_mapping[kv_len - 1: kv_len - 1 + M],
                torch.tensor([0, M], device=device, dtype=torch.int32),
                torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32),
                bt, M, kv_len - 1 + M, gdn_slot,
                collect_aux_from=kv_len - 1, gdn_checkpoint=True))
        seg2.setdefault("verify.target_forward", []).append(ms)

        ms, res = timeit(lambda: _accept(draft_tokens, vlogits, N))
        seg2.setdefault("accept", []).append(ms)
        accepted, bonus = res
        # 去 rollback：不再 copy_ 回 pool（原 0.406ms/step 的 DtoD 已消除）。
        # 只记录 accepted 供下一步 verify 读 checkpoint[accepted]（纯 Python，无 GPU 拷贝）。
        accepted_prev = accepted
        seg2.setdefault("rollback", []).append(0.0)

        anchor = bonus
        kv_len += accepted + 1

    print("\n=== 分段计时（ms/step，均值 over %d iters）===" % N_ITER)
    for k in ["draft.aux_gather", "draft.combine_fc", "draft.precompute_ctx_kv",
              "draft.query_embed", "draft.forward_5L", "draft.select_tokens",
              "verify.target_forward", "accept", "rollback"]:
        v = seg2[k]
        print(f"  {k:26s} {sum(v)/len(v):8.3f} ms   (min {min(v):.3f} max {max(v):.3f})")
    draft_total = sum(sum(seg2[k])/len(seg2[k]) for k in
                      ["draft.aux_gather", "draft.combine_fc", "draft.precompute_ctx_kv",
                       "draft.query_embed", "draft.forward_5L", "draft.select_tokens"])
    print(f"  {'draft TOTAL':26s} {draft_total:8.3f} ms")
    print(f"  {'accept+rollback':26s} {sum(seg2['accept'])/len(seg2['accept'])+sum(seg2['rollback'])/len(seg2['rollback']):8.3f} ms")

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
