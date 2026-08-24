"""逐中间量对比 GDN：HF 模块 vs micro kernel。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from kernel.rmsnorm import rmsnorm1

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "The capital of France is"


def cmp(name, a, b):
    a = a.float().flatten(); b = b.float().flatten()
    d = (a - b).abs().max().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"  {name:14s} max_diff={d:10.5f} cos={cos:10.6f} hf_norm={a.norm().item():9.4f} micro_norm={b.norm().item():9.4f}")


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
    h = text.embed_tokens(input_ids)[0]  # [L,1024]
    in_ln = text.layers[0].input_layernorm
    la = text.layers[0].linear_attn
    with torch.no_grad():
        normed = in_ln(h.unsqueeze(0))[0]  # [L,1024]

    # ---- HF 中间量 ----
    with torch.no_grad():
        mixed_qkv = la.in_proj_qkv(normed.unsqueeze(0))  # [1,L,6144]
        z = la.in_proj_z(normed.unsqueeze(0))            # [1,L,2048]
        b = la.in_proj_b(normed.unsqueeze(0))            # [1,L,16]
        a = la.in_proj_a(normed.unsqueeze(0))            # [1,L,16]
        # conv
        from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn
        mqt = mixed_qkv.transpose(1, 2)  # [1,6144,L]
        conv_out = causal_conv1d_fn(mqt, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation)
        conv_out = conv_out.transpose(1, 2)  # [1,L,6144]
        beta = b.sigmoid()
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
        # split qkv
        kd, vd = la.key_dim, la.value_dim
        q, k, v = torch.split(conv_out, [kd, kd, vd], dim=-1)
        q = q.reshape(L, -1, la.head_k_dim); k = k.reshape(L, -1, la.head_k_dim); v = v.reshape(L, -1, la.head_v_dim)
        # recurrent (exact)
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
        core, _ = torch_recurrent_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g=g, beta=beta,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        core = core[0]  # [L,16,128]
        core2d = core.reshape(-1, la.head_v_dim)
        z2d = z.reshape(-1, la.head_v_dim)
        normed_gdn = la.norm(core2d, z2d)  # [L*16, 128]
        out = la.out_proj(normed_gdn.reshape(L, -1))  # [L,1024]

    # ---- micro 中间量 ----
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
    eng.prefill_runner.adapter.on_prefill_batch(
        [type('S', (), {'seq_id': sid, 'prefill_done': 0})()], eng.prefill_runner)
    block = eng.adapter.blocks(eng.model)[0]
    mla = block.linear_attn
    from kernel.gemv import gemv_or_matmul
    from models.qwen3_5.adapter import (gdn_gbeta, _gdn_conv_prefill_kernel,
                                        _gdn_recurrent_prefill_kernel, _gdn_norm_gated_kernel)
    import triton
    H, DK, DV = 16, 128, 128
    conv_dim = 6144
    with torch.no_grad():
        m = rmsnorm1(h, block._in_ln_w, block._in_ln_eps)
        cmp("normed", m, normed)
        qkv = torch.empty(L, conv_dim, dtype=m.dtype, device=eng.device)
        gemv_or_matmul(m, mla._qkv_w, qkv, "MICRO_GEMV_GDN")
        cmp("qkv_raw", qkv, mixed_qkv[0])
        z_m = torch.empty(L, H*DV, dtype=m.dtype, device=eng.device)
        gemv_or_matmul(m, mla._z_w, z_m, "MICRO_GEMV_GDN")
        cmp("z", z_m, z[0])
        b_m = torch.empty(L, H, dtype=m.dtype, device=eng.device)
        a_m = torch.empty(L, H, dtype=m.dtype, device=eng.device)
        gemv_or_matmul(m, mla._b_w, b_m, "MICRO_GEMV_GDN")
        gemv_or_matmul(m, mla._a_w, a_m, "MICRO_GEMV_GDN")
        cmp("b", b_m, b[0]); cmp("a", a_m, a[0])
        g_m = torch.empty(L, H, dtype=m.dtype, device=eng.device)
        beta_m = torch.empty(L, H, dtype=m.dtype, device=eng.device)
        gdn_gbeta(a_m, b_m, mla._a_log, mla._dt_bias, g_m, beta_m)
        cmp("g", g_m, g[0]); cmp("beta", beta_m, beta[0])
        # conv（清零状态池，避免 warmup 残留）
        conv_state = eng.prefill_runner._gdn_conv_state_pool
        state = eng.prefill_runner._gdn_state_pool
        conv_state.zero_(); state.zero_()
        n_gdn = eng.prefill_runner.adapter._n_gdn
        gdn_l = mla._gdn_layer_idx
        _gdn_conv_prefill_kernel[(triton.cdiv(conv_dim, 512), 1)](
            qkv, mla._conv_w, conv_state, cu_q, eng.prefill_runner._gdn_prefill_seq_idx[:1],
            conv_dim, n_gdn, gdn_l, K=4, BLOCK_C=512)
        cmp("qkv_conv", qkv, conv_out[0])
        # recurrent
        o = torch.empty(L, H*DV, dtype=m.dtype, device=eng.device)
        _gdn_recurrent_prefill_kernel[(1, H)](
            qkv, g_m, beta_m, state, o, cu_q, eng.prefill_runner._gdn_prefill_seq_idx[:1],
            H=H, DK=DK, DV=DV, N_GDN=n_gdn, GDN_L=gdn_l, SCALE=DK**-0.5, BLOCK_D=128)
        cmp("recurrent_o", o, core.reshape(L, -1))
        # norm_gated
        og = torch.empty(L, H*DV, dtype=m.dtype, device=eng.device)
        _gdn_norm_gated_kernel[(L, H)](o, z_m, mla._norm_w, og, H=H, DV=DV,
                                       eps=mla._norm_eps, BLOCK_D=128)
        cmp("norm_gated", og, normed_gdn)
        # out_proj
        out_m = torch.empty(L, 1024, dtype=m.dtype, device=eng.device)
        gemv_or_matmul(og, mla._o_w, out_m, "MICRO_GEMV_GDN")
        cmp("out_proj", out_m, out)


if __name__ == "__main__":
    main()
