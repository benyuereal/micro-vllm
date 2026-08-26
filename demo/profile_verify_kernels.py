"""torch.profiler 分解 verify forward 的 kernel 级耗时（稳态，编译已 warm）。"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from torch.profiler import profile, ProfilerActivity
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)


@torch.inference_mode()
def main():
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096,
                          spec_decode=True, draft_model_path=DRAFT,
                          num_speculative_tokens=7)
    ctrl = eng._spec_controller
    device = eng.device
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

    # warm 编译
    for _ in range(2):
        dt = ctrl._draft_propose(anchor, kv_len)
        vid = torch.cat([torch.tensor([anchor], device=device), dt])
        M = 8
        vpos = torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64)
        vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
        vcu_q = torch.tensor([0, M], device=device, dtype=torch.int32)
        vcu_k = torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32)
        vl = ctrl._forward(vid, vpos, vslot, vcu_q, vcu_k, bt, M, kv_len - 1 + M,
                                  gdn_slot, collect_aux_from=kv_len - 1, gdn_checkpoint=True)
        ctrl._gdn_rollback(gdn_slot, 0)
        anchor = int(vl[0].argmax()); kv_len += 1

    # profile 一次 verify
    dt = ctrl._draft_propose(anchor, kv_len)
    vid = torch.cat([torch.tensor([anchor], device=device), dt])
    M = 8
    vpos = torch.arange(kv_len - 1, kv_len - 1 + M, device=device, dtype=torch.int64)
    vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
    vcu_q = torch.tensor([0, M], device=device, dtype=torch.int32)
    vcu_k = torch.tensor([0, kv_len - 1 + M], device=device, dtype=torch.int32)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            vl = ctrl._forward(vid, vpos, vslot, vcu_q, vcu_k, bt, M, kv_len - 1 + M,
                                      gdn_slot, collect_aux_from=kv_len - 1, gdn_checkpoint=True)
            ctrl._gdn_rollback(gdn_slot, 0)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
