"""eager 模式下单层 decode 的 kernel timeline。
目的：对比 eager(13 kernel 串行执行) vs graph(replay) 步时间，判断 graph 路径是否有 SM 空窗。
- 若 eager ≈ graph：graph 也是串行，SM 空窗存在 → persistent kernel 有 ROI
- 若 eager >> graph：graph 有 overlap，SM 空窗被 overlap 填了 → persistent ROI 小
"""
import sys, torch, os, time
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
import core.layer.model_graph as mg

# eager forward: 调 decode() 而非 graph replay，block_table 从 cache_manager 取
_orig_forward = mg.ModelGraphRunner.forward
def eager_forward(self, input_ids, cache_manager, batch_size):
    if input_ids is None:
        input_ids = self._input_ids[:batch_size]
    bt = cache_manager._block_table_buffer
    return self.decode(input_ids, batch_size, cache_manager, bt)
mg.ModelGraphRunner.forward = eager_forward

engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
engine.add_request("请详细解释 Transformer 架构中多头自注意力机制的完整计算流程。", max_tokens=120, temperature=0.0)

# prefill 走 graph（恢复 forward）
mg.ModelGraphRunner.forward = _orig_forward
while True:
    b, bt = engine.get_next_batch()
    if bt == "waiting" or not b: time.sleep(0.001); continue
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    if bt == "prefill": break

# 切 eager 跑 decode（先 warmup 20 步不计时）
mg.ModelGraphRunner.forward = eager_forward
for _ in range(20):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
torch.cuda.synchronize()

# 计时 80 步
ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
cnt = 0
ev0.record()
for _ in range(80):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    cnt += 1
ev1.record(); torch.cuda.synchronize()
eager_us = ev0.elapsed_time(ev1) / cnt * 1000
print(f"EAGER  {eager_us:.1f} us/step  ({cnt} steps)")
print(f"GRAPH  8908 us/step (基准)")
print(f"ratio  eager/graph = {eager_us/8908:.2f}x")

# profile 一次拿 kernel 时间线
from torch.profiler import profile, ProfilerActivity
from collections import defaultdict
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
torch.cuda.synchronize()

ker = defaultdict(lambda: [0.0, 0])
for ev in prof.events():
    if hasattr(ev, 'cuda_time_total') and ev.cuda_time_total > 0 and ev.name and 'memcpy' not in ev.name.lower():
        ker[ev.name][0] += ev.cuda_time_total / 1000.0
        ker[ev.name][1] += 1
total = sum(v[0] for v in ker.values())
print(f"\n=== eager 20 步 kernel 总时间 {total:.0f} us ({total/20:.1f} us/step) ===")
print(f"{'kernel':55s} {'us':>8s} {'cnt':>5s} {'us/step':>8s}")
for name, (t, c) in sorted(ker.items(), key=lambda x: -x[1][0])[:20]:
    print(f"{name[:55]:55s} {t:8.1f} {c:5d} {t/20:8.2f}")
