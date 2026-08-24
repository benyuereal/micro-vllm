"""对比 HF chunked vs HF recurrent gated delta rule（相同输入），看算法差异。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn, torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule)

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
        q = q.unsqueeze(0); k = k.unsqueeze(0); v = v.unsqueeze(0)
        core_recur, _ = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        core_chunk, _ = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    d = (core_recur.float() - core_chunk.float()).abs()
    cos = F.cosine_similarity(core_recur.float().flatten(), core_chunk.float().flatten(), dim=0)
    print(f"chunked vs recurrent (same input): max_diff={d.max().item():.6f} cos={cos.item():.6f}")
    print(f"  recur_norm={core_recur.float().norm().item():.5f} chunk_norm={core_chunk.float().norm().item():.5f}")
    # 逐 token
    for t in range(L):
        print(f"  t={t}: max_diff={(core_recur[0,t].float()-core_chunk[0,t].float()).abs().max().item():.6f}")


if __name__ == "__main__":
    main()
