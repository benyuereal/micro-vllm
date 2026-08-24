"""Profile 单用户 decode 步：per-kernel 耗时聚合，定位 GDN vs full attention 占比。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.profiler import profile, ProfilerActivity
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
IN_TOK = 256
OUT_TOK = 768

def make_prompt():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < IN_TOK:
        ids += tok.encode(" data")
    return tok.decode(ids[:IN_TOK])

def run(eng, prompt, n):
    eng.add_request(prompt, n, temperature=0.01, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)

eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
prompt = make_prompt()
run(eng, prompt, 8)  # warmup + graph capture

# profile 一段 decode（跳过 prefill，只抓 decode step）
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    run(eng, prompt, OUT_TOK)
    torch.cuda.synchronize()

ka = prof.key_averages()
rows = []
for e in ka:
    if e.device_type == torch.autograd.DeviceType.CUDA or "cuda_time" in dir(e):
        t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if t > 0:
            rows.append((t, e.count, e.key))
rows.sort(reverse=True)
total = sum(r[0] for r in rows)
print(f"\n=== top kernels (self CUDA time, us) total={total/1000:.1f}ms ===")
for t, c, k in rows[:30]:
    print(f"  {t/1000:8.2f}ms x{c:4d}  {k[:80]}")
