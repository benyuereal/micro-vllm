"""量化 attention() 剩余 fragment op（new_pos cast 链 / bt.contiguous）成本。

tag:
  full    : 现在线上路径（new_pos 有 .long()→.to(int32) 冗余 cast，bt .contiguous()）
  clean   : new_pos 简化为 (cache_lens-1).clamp(min=0)（int32 全程，零 cast）
  nobt    : clean + bt 用 view（block_table[:bs] 不 .contiguous()）

输出：RESULT {tag} {us} us/step
"""
import sys, os, torch, time
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
import core.layer.model_graph as mg
import models.deepseek.adapter as adp

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
H = 2048; dtype = torch.bfloat16

_orig_cap = mg.ModelGraphRunner.capture
def _cap1(self, cm, batch_sizes=None): return _orig_cap(self, cm, batch_sizes=[1])
mg.ModelGraphRunner.capture = _cap1

_o_attn = adp.DeepSeekAdapter.attention
def attn_patch(self, x_normed, block, layer_idx, bs, graph, cm, block_table):
    import torch.nn.functional as F
    from kernel.pre_mla import get_premla_persistent_kernel
    from kernel.mla import _get_kernel as _get_mla_kernel
    attn = block.self_attn
    k_cache, v_cache = cm.get(layer_idx)
    cache_lens = cm._cache_seqlens_buffer[:bs]
    if TAG == "full":
        new_pos = (cache_lens - 1).long().clamp(min=0)
    else:  # clean, nobt
        new_pos = (cache_lens - 1).clamp(min=0)            # int32 全程，省 .long()+.to(int32)
    max_len = graph._cur_bucket_maxlen
    block_size = cm.block_size
    cos, sin = self._rope_tables(graph)
    x16 = graph._x16[:bs]
    if TAG in ("full", "clean"):
        bt = block_table[:bs].contiguous()
    else:  # nobt
        bt = block_table[:bs]                              # view，省 .contiguous() 节点
    k_pers, q_out_p, q_pe = get_premla_persistent_kernel(
        bs, self._hidden, self._num_heads, self._q_head, self._qk_rope,
        self._qk_nope, self._kv_lora_rank, self._latent_dim, block_size,
        bt.shape[1], k_cache.shape[0], cos.shape[0], graph.dtype)
    A_in = k_pers(attn._q_w, attn._q_b, cos, sin, attn._kva_w, attn._kva_b,
                  attn._kvb_w_kn_t, graph._absorb_idx[:bs * self._num_heads],
                  x16, q_out_p, q_pe, bt, new_pos.to(torch.int32), k_cache, v_cache)
    A_in = A_in.reshape(bs, self._num_heads, self._kv_lora_rank)
    cos_k = cos[:max_len]; sin_k = sin[:max_len]
    Latent_flat = k_cache.reshape(-1, 1, self._latent_dim)
    n_slots = k_cache.shape[0] * block_size
    kernel = _get_mla_kernel(
        bs, self._num_heads, max_len, self._kv_lora_rank, self._qk_rope,
        self._qk_nope, self._v_head, block_size, graph._ds_softmax_scale,
        graph.dtype, n_slots, block_N=64, num_split=4)
    attn_out = kernel(
        A_in, q_pe, Latent_flat, block_table[:bs], cache_lens,
        attn._kva_ln_w, attn._kvb_w_v, cos_k, sin_k)
    attn_out = attn_out.reshape(bs, self._num_heads * self._v_head)
    return F.linear(attn_out, attn._o_w, attn._o_b)

adp.DeepSeekAdapter.attention = attn_patch

engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
engine.add_request("请详细解释 Transformer 架构中多头自注意力机制的完整计算流程。", max_tokens=120, temperature=0.0)
while True:
    b, bt = engine.get_next_batch()
    if bt == "waiting" or not b: time.sleep(0.001); continue
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    if bt == "prefill": break
for _ in range(20):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
torch.cuda.synchronize()
ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
cnt=0; ev0.record()
for _ in range(80):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    cnt+=1
ev1.record(); torch.cuda.synchronize()
print(f"RESULT {TAG} {ev0.elapsed_time(ev1)/cnt*1000:.1f} us/step")
