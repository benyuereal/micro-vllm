"""实验：MLA decode kernel 的 num_split 参数扫描。

发现：MLA split 阶段 grid=(1,1,num_split)，当前 num_split=4 → 只用 4 个 SM（92 中 88 空闲）。
MLA 实测 38.4us/层 vs HBM floor 3.1us = 10-12x gap，疑似 SM-occupancy-bound 非 memory-bound。
增大 num_split → 更多 SM 并行读 Latent，可能压低 MLA 耗时。

本脚本在 graph 路径下扫 num_split ∈ {4, 8, 16, 32, 64}，量稳态步时间 + token 正确性。
注意：num_split 改变会触发新 kernel 编译（cache key 含 num_split），且 combine 阶段
P_partial buffer 变大（batch*h_q*num_split*kv_lora）。

用法: python3 tests/prof_mla_split.py
"""
import sys, os, torch, time
sys.path.insert(0, "/models/micro-vllm")
import core.layer.model_graph as mg
import models.deepseek.adapter as adp

# patch num_split: 在 attention() 调 _get_mla_kernel 前替换参数
SPLIT = int(os.environ.get("MLA_SPLIT", "4"))

_orig_attn = adp.DeepSeekAdapter.attention
def patched_attn(self, x_normed, block, layer_idx, bs, graph, cm, block_table):
    import torch.nn.functional as F
    from kernel.pre_mla import get_premla_persistent_kernel
    from kernel.mla import _get_kernel as _get_mla_kernel
    attn = block.self_attn
    k_cache, v_cache = cm.get(layer_idx)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    new_pos = (cache_lens - 1).long().clamp(min=0)
    max_len = graph._cur_bucket_maxlen
    block_size = cm.block_size
    cos, sin = self._rope_pool(graph, k_cache.device)
    cos_q = cos[new_pos].to(graph.dtype); sin_q = sin[new_pos].to(graph.dtype)
    x16 = graph._x16[:bs]; bt = block_table[:bs].contiguous()
    k_pers, q_out_p = get_premla_persistent_kernel(
        bs, self._hidden, self._num_heads, self._q_head, self._qk_rope,
        self._qk_nope, self._kv_lora_rank, self._latent_dim, block_size,
        bt.shape[1], k_cache.shape[0], graph.dtype)
    A_in = k_pers(attn._q_w, attn._q_b, cos_q, sin_q, attn._kva_w, attn._kva_b,
                  attn._kvb_w_kn_t, graph._absorb_idx[:bs * self._num_heads],
                  x16, q_out_p, bt, new_pos.to(torch.int32), k_cache, v_cache)
    A_in = A_in.reshape(bs, self._num_heads, self._kv_lora_rank)
    q_pe = q_out_p[:, :, 0, self._qk_nope:].contiguous()
    k_pos = torch.arange(max_len, device=k_cache.device)
    cos_k = cos[k_pos].contiguous(); sin_k = sin[k_pos].contiguous()
    Latent_flat = k_cache.reshape(-1, 1, self._latent_dim).contiguous()
    n_slots = k_cache.shape[0] * block_size
    kernel = _get_mla_kernel(
        bs, self._num_heads, max_len, self._kv_lora_rank, self._qk_rope,
        self._qk_nope, self._v_head, block_size, graph._ds_softmax_scale,
        graph.dtype, n_slots, block_N=64, num_split=SPLIT)
    attn_out = kernel(
        A_in, q_pe, Latent_flat, block_table[:bs].contiguous(),
        cache_lens.to(torch.int32).contiguous(),
        attn._kva_ln_w, attn._kvb_w_v, cos_k, sin_k)
    attn_out = attn_out.reshape(bs, self._num_heads * self._v_head)
    return F.linear(attn_out, attn._o_w, attn._o_b)
adp.DeepSeekAdapter.attention = patched_attn

# 只 capture bs=1
_orig_cap = mg.ModelGraphRunner.capture
def _cap1(self, cm, batch_sizes=None): return _orig_cap(self, cm, batch_sizes=[1])
mg.ModelGraphRunner.capture = _cap1

from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
engine.add_request("请详细解释 Transformer 架构中多头自注意力机制的完整计算流程。", max_tokens=140, temperature=0.0)
while True:
    b, bt = engine.get_next_batch()
    if bt == "waiting" or not b: time.sleep(0.001); continue
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    if bt == "prefill": break
# 收 token
ids = []
for _ in range(25):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    ids.append(ctx.sequences[0].output_ids[-1])
torch.cuda.synchronize()

# 计时
ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
cnt=0; ev0.record()
for _ in range(80):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    cnt+=1
ev1.record(); torch.cuda.synchronize()
us = ev0.elapsed_time(ev1)/cnt*1000
tok_s = 1e6/us
print(f"RESULT split={SPLIT} {us:.1f} us/step  {tok_s:.1f} tok/s")
print(f"IDS=" + ",".join(str(i) for i in ids[:20]))
