"""prefill 逐层计时：定位 1084ms 花在哪（GEMM 反量化 vs GDN vs flash vs 其他）。"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine
from models.base import PrefillMeta

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
    meta = PrefillMeta(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, position_ids=positions,
                       slot_mapping=slot_mapping[:P], block_table=bt, n_seqs=1,
                       max_seqlen_q=P, max_seqlen_k=P)

    # warm
    ctrl._forward(prompt_ids, positions, slot_mapping[:P], cu_q, cu_k, bt,
                         P, P, gdn_slot, collect_aux_from=0, gdn_checkpoint=False)
    torch.cuda.synchronize()

    # 逐层计时
    h = ctrl.embed(prompt_ids)
    layer_ms = []
    for li in range(ctrl.num_layers):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = ctrl.adapter.prefill(ctrl.blocks[li], h, li, eng.prefill_runner,
                                 eng.cache_manager, meta)
        torch.cuda.synchronize()
        layer_ms.append((time.perf_counter() - t0) * 1000)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = ctrl.final_norm(h)
    out = ctrl.lm_head(h)
    torch.cuda.synchronize()
    tail = (time.perf_counter() - t0) * 1000

    gdn = [m for m, li in zip(layer_ms, range(ctrl.num_layers)) if ctrl.blocks[li]._is_gdn]
    full = [m for m, li in zip(layer_ms, range(ctrl.num_layers)) if not ctrl.blocks[li]._is_gdn]
    print(f"\n=== prefill 逐层 (P={P}) ===")
    print(f"  GDN 层 x{len(gdn)}: 均值 {sum(gdn)/len(gdn):.2f} ms, 合计 {sum(gdn):.1f} ms")
    print(f"  full-attn 层 x{len(full)}: 均值 {sum(full)/len(full):.2f} ms, 合计 {sum(full):.1f} ms")
    print(f"  final_norm+lm_head: {tail:.2f} ms")
    print(f"  层合计: {sum(layer_ms):.1f} ms + tail {tail:.1f} = {sum(layer_ms)+tail:.1f} ms")
    print(f"  最慢 5 层: {sorted(layer_ms, reverse=True)[:5]}")

    eng.cache_manager.free(seq_id)
    ctrl._gdn_free(gdn_slot)


if __name__ == "__main__":
    main()
