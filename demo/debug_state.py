"""对比 prefill 后的 GDN 递归状态：HF chunked vs micro recurrent。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn, torch_chunk_gated_delta_rule)

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
    h = text.embed_tokens(input_ids)[0]
    la = text.layers[0].linear_attn
    with torch.no_grad():
        normed = text.layers[0].input_layernorm(h.unsqueeze(0))[0]
        mixed_qkv = la.in_proj_qkv(normed.unsqueeze(0))
        b = la.in_proj_b(normed.unsqueeze(0))
        a = la.in_proj_a(normed.unsqueeze(0))
        mqt = mixed_qkv.transpose(1, 2)
        conv_out = causal_conv1d_fn(mqt, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation).transpose(1, 2)
        beta = b.sigmoid()
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
        kd, vd = la.key_dim, la.value_dim
        q, k, v = torch.split(conv_out, [kd, kd, vd], dim=-1)
        q = q.reshape(L, -1, la.head_k_dim); k = k.reshape(L, -1, la.head_k_dim); v = v.reshape(L, -1, la.head_v_dim)
        # HF chunked prefill → final state
        _, state_hf = torch_chunk_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g=g, beta=beta,
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        # state_hf: [1, 16, 128, 128] fp32

    # micro prefill → state
    from core.engine import InferenceEngine
    from models.base import PrefillMeta
    eng = InferenceEngine(MODEL, max_batch_size=8, max_prefill_tokens=4096)
    cm = eng.cache_manager; bs = cm.block_size; sid = 99999
    cm.alloc(sid, L)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    pos = torch.arange(L, device=eng.device)
    bt = cm._block_table_buffer[:1]; cm.cache_batch_data([sid], [L])
    abs_pos = torch.arange(L, device=eng.device)
    slot = (bt[0, abs_pos // bs] * bs + abs_pos % bs).to(torch.int32)
    meta = PrefillMeta(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, position_ids=pos,
                       slot_mapping=slot, block_table=bt, n_seqs=1, max_seqlen_q=L, max_seqlen_k=L)
    # 清零 + 分配 slot
    eng.prefill_runner._gdn_state_pool.zero_()
    eng.prefill_runner._gdn_conv_state_pool.zero_()
    dummy = type('S', (), {'seq_id': sid, 'prefill_done': 0})()
    eng.prefill_runner.adapter.on_prefill_batch([dummy], eng.prefill_runner)
    slot = dummy._gdn_slot
    print(f"allocated slot={slot}")
    block = eng.adapter.blocks(eng.model)[0]
    from kernel.rmsnorm import rmsnorm1
    with torch.no_grad():
        normed_m = rmsnorm1(h, block._in_ln_w, block._in_ln_eps)
        eng.prefill_runner.adapter._gdn_forward(
            block.linear_attn, normed_m, eng.prefill_runner, L,
            is_decode=False, cu_seqlens=cu_q,
            seq_idx=eng.prefill_runner._gdn_prefill_seq_idx[:1])
    # micro state: pool[slot, gdn_layer=0, 16, 128, 128]
    state_m = eng.prefill_runner._gdn_state_pool[slot, 0]  # [16,128,128] fp32
    state_hf = state_hf[0]  # [16,128,128]
    d = (state_hf.float() - state_m.float()).abs()
    cos = F.cosine_similarity(state_hf.float().flatten(), state_m.float().flatten(), dim=0)
    print(f"prefill final state: max_diff={d.max().item():.6f} cos={cos.item():.6f}")
    print(f"  hf_norm={state_hf.float().norm().item():.5f} micro_norm={state_m.float().norm().item():.5f}")
    # 逐 head
    for hh in range(16):
        dd = (state_hf[hh].float()-state_m[hh].float()).abs().max().item()
        print(f"  head {hh}: max_diff={dd:.6f}")


if __name__ == "__main__":
    main()
