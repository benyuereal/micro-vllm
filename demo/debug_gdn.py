"""聚焦对比：单层 GDN（layer 0）HF linear_attn vs micro _gdn_forward。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    hf.eval()
    ids = tok.encode(PROMPT, add_special_tokens=True)
    input_ids = torch.tensor([ids], device="cuda:0")
    L = len(ids)

    text = hf.model
    h = text.embed_tokens(input_ids)[0]  # [L, 1024]
    # HF decoder layer: input_layernorm 后再调 linear_attn。用 HF 的 norm 算 normed。
    in_ln = text.layers[0].input_layernorm
    with torch.no_grad():
        normed_hf = in_ln(h.unsqueeze(0))[0]  # [L, 1024]

    # HF layer 0 GDN（喂 normed）
    la = text.layers[0].linear_attn
    with torch.no_grad():
        hf_gdn = la(normed_hf.unsqueeze(0), cache_params=None)  # [1, L, 1024]
    hf_gdn = hf_gdn[0]  # [L, 1024]

    # micro
    from core.engine import InferenceEngine
    from models.base import PrefillMeta
    eng = InferenceEngine(MODEL, max_batch_size=8, max_prefill_tokens=4096)
    cm = eng.cache_manager
    bs = cm.block_size
    sid = 99999
    cm.alloc(sid, L)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    pos = torch.arange(L, device=eng.device)
    bt = cm._block_table_buffer[:1]
    cm.cache_batch_data([sid], [L])
    abs_pos = torch.arange(L, device=eng.device)
    slot = (bt[0, abs_pos // bs] * bs + abs_pos % bs).to(torch.int32)
    meta = PrefillMeta(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, position_ids=pos,
                       slot_mapping=slot, block_table=bt, n_seqs=1,
                       max_seqlen_q=L, max_seqlen_k=L)
    # 填 prefill seq_idx（slot 0）
    eng.prefill_runner.adapter.on_prefill_batch(
        [type('S', (), {'seq_id': sid, 'prefill_done': 0})()], eng.prefill_runner)

    block = eng.adapter.blocks(eng.model)[0]
    from kernel.rmsnorm import rmsnorm1
    with torch.no_grad():
        normed = rmsnorm1(h, block._in_ln_w, block._in_ln_eps)
        # 先验证 normed 与 HF 一致
        nd = (normed.float() - normed_hf.float()).abs()
        print(f"normed: max_diff={nd.max().item():.6f} cos={F.cosine_similarity(normed.float().flatten(), normed_hf.float().flatten(), dim=0).item():.6f}")
        micro_gdn = eng.prefill_runner.adapter._gdn_forward(
            block.linear_attn, normed, eng.prefill_runner, L,
            is_decode=False, cu_seqlens=cu_q, seq_idx=eng.prefill_runner._gdn_prefill_seq_idx[:1])

    # 对比
    d = (hf_gdn.float() - micro_gdn.float()).abs()
    print(f"GDN out: shape hf={tuple(hf_gdn.shape)} micro={tuple(micro_gdn.shape)}")
    print(f"  max_abs_diff={d.max().item():.5f} mean={d.mean().item():.6f}")
    print(f"  cos={F.cosine_similarity(hf_gdn.float().flatten(), micro_gdn.float().flatten(), dim=0).item():.6f}")
    print(f"  hf norm={hf_gdn.float().norm().item():.4f} micro norm={micro_gdn.float().norm().item():.4f}")
    print(f"  hf[0,:6]={hf_gdn[0,:6].tolist()}")
    print(f"  micro[0,:6]={micro_gdn[0,:6].tolist()}")

    # 逐 token 看哪里开始偏
    print("\n逐 token max_abs_diff:")
    for t in range(L):
        print(f"  t={t}: {(hf_gdn[t].float()-micro_gdn[t].float()).abs().max().item():.5f}")


if __name__ == "__main__":
    main()
