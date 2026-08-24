"""逐层对比 prefill hidden states：micro vs HF。定位第一个分叉层。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/models/Qwen3.5-0.8B"
PROMPT = "The capital of France is"

# ---- HF ----
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                          device_map="cuda:0", trust_remote_code=True,
                                          local_files_only=True)
hf.eval()
ids = tok.encode(PROMPT, add_special_tokens=True)
with torch.no_grad():
    out = hf(torch.tensor([ids], device="cuda:0"), output_hidden_states=True)
hf_layers = out.hidden_states  # (emb, L0..L23)
print(f"HF: {len(hf_layers)} hidden states, last-token shape {hf_layers[0][0,-1].shape}")
hf_last = [hs[0, -1].float() for hs in hf_layers]  # [hidden] each

# ---- micro ----
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
pad = eng.prefill_runner.adapter
pad._dbg = []
eng.add_request(PROMPT, 1, temperature=0.0, top_p=1.0)
b, bt = eng.get_next_batch()
assert bt == "prefill", f"expected prefill, got {bt}"
ctx = BatchInferenceContext(len(b), bt, b)
eng.step(ctx)
eng.collect(ctx)
eng.update_sequences(ctx.sequences)
dbg = pad._dbg
print(f"micro: {len(dbg)} layers captured")

# 对比：HF layer i 的 last-token vs micro layer i 的 out last-token
# micro dbg[i] = (layer_idx, in_h, out_h)，out_h 是第 i 层输出 [T, hidden]
maxdiff = []
for i, (li, in_h, out_h) in enumerate(dbg):
    # HF hidden_states[i+1] = 第 i 层输出
    hf_i = hf_last[i + 1]
    mc_i = out_h[-1].float()
    d = (hf_i - mc_i).abs().max().item()
    rel = d / (hf_i.abs().max().item() + 1e-6)
    maxdiff.append((i, li, d, rel))
    flag = "  <<<" if d > 0.5 else ""
    print(f"layer {i:2d} (type {li}): maxdiff={d:.4f} rel={rel:.4f}{flag}")

# 也对比 embedding 输出（HF hidden_states[0] vs micro 第0层 in_h）
emb_d = (hf_last[0] - dbg[0][1][-1].float()).abs().max().item()
print(f"embedding last-token maxdiff={emb_d:.4f}")
