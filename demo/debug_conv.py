"""隔离 conv kernel：相同输入，对比 micro conv vs HF conv（bf16 和 fp32）。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

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
        mixed_qkv = la.in_proj_qkv(normed.unsqueeze(0))  # [1,L,6144] bf16
        # HF conv (bf16, 内部 to(weight.dtype)=bf16)
        mqt = mixed_qkv.transpose(1, 2)
        conv_bf16 = causal_conv1d_fn(mqt, la.conv1d.weight.squeeze(1), la.conv1d.bias, activation=la.activation).transpose(1, 2)[0]
        # HF conv fp32 参考：input [1, 6144, L]，左 pad 3
        w32 = la.conv1d.weight.squeeze(1).float()  # [6144, 4]
        x32 = mixed_qkv[0].float().transpose(0, 1)  # [6144, L]
        x32p = F.pad(x32, (3, 0)).unsqueeze(0)      # [1, 6144, L+3]
        conv_fp32 = F.conv1d(x32p, w32.unsqueeze(1), None, padding=0, groups=6144)[:, :, :L].squeeze(0)
        conv_fp32 = F.silu(conv_fp32).transpose(0, 1)  # [L, 6144]

    # micro conv
    from core.engine import InferenceEngine
    eng = InferenceEngine(MODEL, max_batch_size=8, max_prefill_tokens=4096)
    dev = eng.device
    from models.qwen3_5.adapter import _gdn_conv_prefill_kernel
    import triton
    conv_dim = 6144
    qkv = mixed_qkv[0].contiguous()  # [L,6144] 相同输入
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
    seq_idx = torch.zeros(1, dtype=torch.int32, device=dev)
    conv_state = torch.zeros(1, 18, 3, conv_dim, dtype=qkv.dtype, device=dev)
    _gdn_conv_prefill_kernel[(triton.cdiv(conv_dim, 512), 1)](
        qkv, la.conv1d.weight.squeeze(1).contiguous(), conv_state, cu_q, seq_idx,
        conv_dim, K=4, BLOCK_C=512)

    def cmp(name, a, b):
        a = a.float().flatten(); b = b.float().flatten()
        d = (a - b).abs().max().item()
        cos = F.cosine_similarity(a, b, dim=0).item()
        print(f"  {name:16s} max_diff={d:10.5f} cos={cos:10.6f}")

    cmp("micro vs hf_bf16", qkv, conv_bf16)
    cmp("micro vs hf_fp32", qkv, conv_fp32)
    cmp("hf_bf16 vs hf_fp32", conv_bf16, conv_fp32)
    # 看第一个 token（纯 zero-pad，无 state 依赖）
    print(f"\n  t=0: micro={qkv[0,:4].tolist()}")
    print(f"        hf_bf16={conv_bf16[0,:4].tolist()}")
    print(f"        hf_fp32={conv_fp32[0,:4].tolist()}")
    # 看 conv_state 末尾（应存最后 3 个 pre-act 输入）
    print(f"\n  conv_state[0,0,0,:4]={conv_state[0,0,0,:4].tolist()}  (应=qkv[0,:4] pre-act)")
    print(f"  conv_state[0,0,1,:4]={conv_state[0,0,1,:4].tolist()}  (应=qkv[1,:4])")
    print(f"  conv_state[0,0,2,:4]={conv_state[0,0,2,:4].tolist()}  (应=qkv[2,:4])")
    print(f"  qkv pre-act[0,:4] (已被覆盖，无法直接看)")


if __name__ == "__main__":
    main()
