"""隔离 recurrent kernel：喂相同 conv 输出，对比 micro recurrent vs HF recurrent。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn, torch_recurrent_gated_delta_rule)

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
        z = la.in_proj_z(normed.unsqueeze(0))
        b = la.in_proj_b(normed.unsqueeze(0))
        a = la.in_proj_a(normed.unsqueeze(0))
        mqt = mixed_qkv.transpose(1, 2)
        conv_out = causal_conv1d_fn(mqt, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation).transpose(1, 2)
        beta = b.sigmoid()
        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)
        kd, vd = la.key_dim, la.value_dim
        q, k, v = torch.split(conv_out, [kd, kd, vd], dim=-1)
        q = q.reshape(L, -1, la.head_k_dim); k = k.reshape(L, -1, la.head_k_dim); v = v.reshape(L, -1, la.head_v_dim)
        core_hf, _ = torch_recurrent_gated_delta_rule(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g=g, beta=beta,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        core_hf = core_hf[0].reshape(L, -1)  # [L, 2048]

    # micro recurrent，喂 HF 的 conv_out（相同输入）
    from core.engine import InferenceEngine
    eng = InferenceEngine(MODEL, max_batch_size=8, max_prefill_tokens=4096)
    dev = eng.device
    from models.qwen3_5.adapter import _gdn_recurrent_prefill_kernel
    import triton
    H, DK, DV = 16, 128, 128
    conv_dim = 6144
    qkv = conv_out[0].contiguous()  # [L, 6144] 用 HF 的 conv 输出
    g_m = g[0].contiguous(); beta_m = beta[0].contiguous()
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
    seq_idx = torch.zeros(1, dtype=torch.int32, device=dev)
    state = torch.zeros(1, 18, H, DK, DV, dtype=torch.float32, device=dev)
    o = torch.empty(L, H*DV, dtype=qkv.dtype, device=dev)
    _gdn_recurrent_prefill_kernel[(1, H)](
        qkv, g_m, beta_m, state, o, cu_q, seq_idx,
        H=H, DK=DK, DV=DV, N_GDN=18, GDN_L=0, SCALE=DK**-0.5, BLOCK_D=128)
    d = (core_hf.float() - o.float()).abs()
    print(f"recurrent (same conv input): max_diff={d.max().item():.6f} "
          f"cos={F.cosine_similarity(core_hf.float().flatten(), o.float().flatten(), dim=0).item():.6f}")
    print(f"  hf_norm={core_hf.float().norm().item():.5f} micro_norm={o.float().norm().item():.5f}")
    print(f"  hf[0,:6]={core_hf[0,:6].tolist()}")
    print(f"  micro[0,:6]={o[0,:6].tolist()}")
    # 逐 token
    for t in range(L):
        print(f"  t={t}: max_diff={(core_hf[t].float()-o[t].float()).abs().max().item():.6f}")


if __name__ == "__main__":
    main()
