"""decode 第 1 步逐层 residual stream 对比：micro (eager) vs HF (forward hooks)。"""
import os, sys
sys.path.insert(0, "/tmp/micro-vllm-w8a16")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/models/Qwen3.5-0.8B"
PROMPT = "The capital of France is"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                          device_map="cuda:0", trust_remote_code=True,
                                          local_files_only=True)
hf.eval()
ids = tok.encode(PROMPT, add_special_tokens=True)
x = torch.tensor([ids], device="cuda:0")

text = hf.model
layer_outs = {}
hooks = []
for i, layer in enumerate(text.layers):
    def mk(i):
        def h(mod, inp, out):
            layer_outs[i] = out.detach().clone()
        return h
    hooks.append(layer.register_forward_hook(mk(i)))

with torch.no_grad():
    out1 = hf(x, use_cache=True)
    past = out1.past_key_values
    first_tok = out1.logits[0, -1].argmax().item()
    x2 = torch.tensor([[first_tok]], device="cuda:0")
    pos = torch.tensor([[len(ids)]], device="cuda:0")
    out2 = hf(x2, past_key_values=past, position_ids=pos, use_cache=True)
hf_dec = {i: layer_outs[i][0, -1].float() for i in layer_outs}
for h in hooks:
    h.remove()
print(f"HF first_tok={first_tok}  n decode layers={len(hf_dec)}")

# ---- micro: prefill + 1 decode step (eager) ----
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
dadapter = eng.graph_runner.adapter
dadapter._dbg_dec = []
eng.add_request(PROMPT, 2, temperature=0.0, top_p=1.0)
b, bt = eng.get_next_batch()
ctx = BatchInferenceContext(len(b), bt, b)
eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
b, bt = eng.get_next_batch()
assert bt == "decode", f"expected decode got {bt}"
ctx = BatchInferenceContext(len(b), bt, b)
eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
dec = dadapter._dbg_dec  # dec[i] = 第 i 层输出 residual stream (i=0..22)
print(f"micro decode layers captured: {len(dec)}")

print("=== decode step 1 per-layer residual stream (last token) ===")
for i in range(min(len(dec), 24)):
    if i not in hf_dec:
        continue
    d = (hf_dec[i] - dec[i][-1].float()).abs().max().item()
    rel = d / (hf_dec[i].abs().max().item() + 1e-6)
    flag = "  <<<" if d > 0.5 else ""
    print(f"layer {i:2d}: maxdiff={d:.4f} rel={rel:.4f}{flag}")
