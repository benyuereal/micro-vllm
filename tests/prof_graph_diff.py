"""graph 路径下 MoE 各子段真实延迟（差分法，子进程隔离）。

差分法：把某段 patch 成 no-op（返回正确 shape），重新 capture graph，量步时间。
差值 = 该段在 graph 路径下的真实贡献。比 in-graph event 可靠（event 时间戳在 replay 下乱序）。

用法：python3 tests/prof_graph_diff.py {tag}
  tag=baseline        → 正常路径
  tag=no_norm         → rmsnorm_residual no-op
  tag=no_gate         → gate_gemv+softmax_topk no-op
  tag=no_routed       → moe_routed_decode no-op
  tag=no_shared       → shared no-op
  tag=no_moe          → 整个 moe_decode no-op
  tag=no_attn         → attention no-op
  tag=fine_tile       → 细化 tile (BLOCK=32)，正常路径
  tag=orig_tile       → 原始 tile (BLOCK=64)，正常路径
  tag=persist         → routed persistent kernel (USE_PERSISTENT_MOE=1)
输出：RESULT {tag} {us} us/step
"""
import sys, os, torch, time, statistics
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
import core.layer.model_graph as mg
import models.deepseek.adapter as adp
import kernel.moe as KM
import kernel.rmsnorm as KR

TAG = sys.argv[1] if len(sys.argv) > 1 else "baseline"
H = 2048; dtype = torch.bfloat16

# 只 capture bs=1
_orig_cap = mg.ModelGraphRunner.capture
def _cap1(self, cm, batch_sizes=None): return _orig_cap(self, cm, batch_sizes=[1])
mg.ModelGraphRunner.capture = _cap1

def zshape(*shape):
    return torch.zeros(*shape, dtype=dtype, device="cuda")

# ---- no-op patches ----
if TAG == "no_norm":
    KR.rmsnorm_residual_gemm = lambda *a, **k: (a[3], a[4])  # 返回 out_normed, out_residual buffer（不计算）
    adp.rmsnorm_residual = KR.rmsnorm_residual_gemm

if TAG == "no_attn":
    _o = adp.DeepSeekAdapter.attention
    def noop_attn(self, x_normed, block, layer_idx, bs, graph, cm, block_table):
        return zshape(bs, H)
    adp.DeepSeekAdapter.attention = noop_attn

# attention 内部细分: no_mla = 跳 MLA decode kernel（attn_out 填 zeros，o_proj 仍跑）
# no_oproj = 跳 o_proj（返回 zeros[bs, hidden]），pre-MLA+MLA 仍跑
if TAG in ("no_mla", "no_oproj"):
    _o_attn = adp.DeepSeekAdapter.attention
    def attn_split(self, x_normed, block, layer_idx, bs, graph, cm, block_table):
        import torch.nn.functional as _F
        attn = block.self_attn
        k_cache, v_cache = cm.get(layer_idx)
        cache_lens = cm._cache_seqlens_buffer[:bs]
        new_pos = (cache_lens - 1).clamp(min=0)
        max_len = graph._cur_bucket_maxlen
        block_size = cm.block_size
        cos, sin = self._rope_pool(graph, k_cache.device)
        x16 = graph._x16[:bs]; bt = block_table[:bs]
        from kernel.pre_mla import get_premla_persistent_kernel
        k_pers, q_out_p, q_pe = get_premla_persistent_kernel(
            bs, self._hidden, self._num_heads, self._q_head, self._qk_rope,
            self._qk_nope, self._kv_lora_rank, self._latent_dim, block_size,
            bt.shape[1], k_cache.shape[0], cos.shape[0], graph.dtype)
        A_in = k_pers(attn._q_w, attn._q_b, cos, sin, attn._kva_w, attn._kva_b,
                      attn._kvb_w_kn_t, graph._absorb_idx[:bs * self._num_heads],
                      x16, q_out_p, q_pe, bt, new_pos, k_cache, v_cache)
        A_in = A_in.reshape(bs, self._num_heads, self._kv_lora_rank)
        if TAG == "no_oproj":
            # pre-MLA + MLA 仍跑，跳 o_proj
            cos_k = cos[:max_len]; sin_k = sin[:max_len]
            Latent_flat = k_cache.reshape(-1, 1, self._latent_dim)
            n_slots = k_cache.shape[0] * block_size
            from kernel.mla import _get_kernel as _get_mla_kernel
            kernel = _get_mla_kernel(
                bs, self._num_heads, max_len, self._kv_lora_rank, self._qk_rope,
                self._qk_nope, self._v_head, block_size, graph._ds_softmax_scale,
                graph.dtype, n_slots, block_N=64, num_split=4)
            attn_out = kernel(
                A_in, q_pe, Latent_flat, block_table[:bs], cache_lens,
                attn._kva_ln_w, attn._kvb_w_v, cos_k, sin_k)
            attn_out = attn_out.reshape(bs, self._num_heads * self._v_head)
            return zshape(bs, self._hidden)
        # no_mla: 跳 MLA decode，o_proj 读 zeros
        attn_out = zshape(bs, self._num_heads * self._v_head)
        return _F.linear(attn_out, attn._o_w, attn._o_b)
    adp.DeepSeekAdapter.attention = attn_split

if TAG == "no_gate":
    # gate_gemv + softmax_topk no-op：返回固定 idx/w
    def noop_moe_decode(x, gate_weight, e_gu, e_d, top_k, n_experts, shared_gu=None, shared_d=None):
        N, Hh = x.shape
        K = top_k
        # 跳过 gate，直接用固定 idx=0..K, w=1
        idx = torch.zeros(N, K, dtype=torch.int32, device="cuda")
        w = torch.ones(N, K, dtype=x.dtype, device="cuda")
        x16 = torch.zeros(N, 16, Hh, dtype=x.dtype, device="cuda"); x16[:,0,:] = x
        out = KM.moe_routed_decode(x, e_gu, e_d, idx, w, x16=x16)
        if shared_gu is not None:
            S_INTER = shared_d.shape[0]
            from kernel.moe import _kernel_cache, _TORCH_TO_TL, shared_gate_up_kernel, shared_down_kernel
            tl_dt = _TORCH_TO_TL[x.dtype]; skey=(N,Hh,S_INTER,x.dtype)
            if skey not in _kernel_cache:
                _kernel_cache[skey] = (shared_gate_up_kernel(N,Hh,S_INTER,tl_dt), shared_down_kernel(N,Hh,S_INTER,tl_dt))
            k_sgu, k_sdn = _kernel_cache[skey]
            sact = k_sgu(x16, shared_gu); k_sdn(sact, shared_d, out)
        return out
    adp.moe_decode = noop_moe_decode
    KM.moe_decode = noop_moe_decode

if TAG == "no_routed":
    _orig_md = KM.moe_decode
    def norouted_moe_decode(x, gate_weight, e_gu, e_d, top_k, n_experts, shared_gu=None, shared_d=None):
        N, Hh = x.shape; K = top_k
        # 仍算 gate（要量的是 routed 部分），跳过 routed，只跑 shared
        tl_dt = KM._TORCH_TO_TL[x.dtype]
        gkey = (N, Hh, n_experts, K, x.dtype)
        if gkey not in KM._kernel_cache:
            KM._kernel_cache[gkey] = (KM.gate_gemv_kernel(N,Hh,n_experts,tl_dt), KM.softmax_topk_kernel(N,n_experts,K,tl_dt))
        k_gv, k_st = KM._kernel_cache[gkey]
        logits = k_gv(x, gate_weight); topk_idx, topk_weight = k_st(logits)
        x16 = torch.zeros(N,16,Hh,dtype=x.dtype,device="cuda"); x16[:,0,:]=x
        out = zshape(N, Hh)  # 跳过 routed
        if shared_gu is not None:
            S_INTER = shared_d.shape[0]; skey=(N,Hh,S_INTER,x.dtype)
            if skey not in KM._kernel_cache:
                KM._kernel_cache[skey] = (KM.shared_gate_up_kernel(N,Hh,S_INTER,tl_dt), KM.shared_down_kernel(N,Hh,S_INTER,tl_dt))
            k_sgu, k_sdn = KM._kernel_cache[skey]
            sact = k_sgu(x16, shared_gu); k_sdn(sact, shared_d, out)
        return out
    adp.moe_decode = norouted_moe_decode; KM.moe_decode = norouted_moe_decode

if TAG == "no_shared":
    _orig_md2 = KM.moe_decode
    def noshared_moe_decode(x, gate_weight, e_gu, e_d, top_k, n_experts, shared_gu=None, shared_d=None):
        N, Hh = x.shape; K = top_k; tl_dt = KM._TORCH_TO_TL[x.dtype]
        gkey = (N,Hh,n_experts,K,x.dtype)
        if gkey not in KM._kernel_cache:
            KM._kernel_cache[gkey] = (KM.gate_gemv_kernel(N,Hh,n_experts,tl_dt), KM.softmax_topk_kernel(N,n_experts,K,tl_dt))
        k_gv, k_st = KM._kernel_cache[gkey]
        logits = k_gv(x, gate_weight); topk_idx, topk_weight = k_st(logits)
        x16 = torch.zeros(N,16,Hh,dtype=x.dtype,device="cuda"); x16[:,0,:]=x
        out = KM.moe_routed_decode(x, e_gu, e_d, topk_idx, topk_weight, x16=x16)
        return out  # 跳过 shared
    adp.moe_decode = noshared_moe_decode; KM.moe_decode = noshared_moe_decode

if TAG == "no_moe":
    def nomoe(x, gate_weight, e_gu, e_d, top_k, n_experts, shared_gu=None, shared_d=None):
        return zshape(x.shape[0], x.shape[1])
    adp.moe_decode = nomoe; KM.moe_decode = nomoe

# routed 内部细分: no_gu = 跳 gate_up（down 读 zeros）；no_down = 跳 down（只 gate_up 写 Act16）
if TAG in ("no_gu", "no_down"):
    _orig_routed = KM.moe_routed_decode
    def routed_split(x, e_gu, e_d, idx, w_gate, x16=None):
        N, Hh = x.shape
        E, TWO_INTER, _ = e_gu.shape; INTER = TWO_INTER // 2; K = idx.shape[1]
        idx_i32 = idx.to(torch.int32); tl_dt = KM._TORCH_TO_TL[x.dtype]
        if x16 is None:
            x16 = torch.zeros(N, 16, Hh, dtype=x.dtype, device=x.device); x16[:, 0, :] = x
        key = (N, Hh, INTER, E, K, x.dtype)
        if key not in KM._kernel_cache:
            KM._kernel_cache[key] = (KM.moe_gate_up_kernel(N,Hh,INTER,E,K,tl_dt), KM.moe_down_kernel(N,Hh,INTER,E,K,tl_dt))
        k_gu, k_dn = KM._kernel_cache[key]
        act16 = torch.zeros(N, K, 16, INTER, dtype=x.dtype, device=x.device)
        if TAG == "no_down":
            k_gu(x16, e_gu, idx_i32, w_gate)   # 只算 gate_up（写 act16，但不读）
            return zshape(N, Hh)
        # no_gu: 跳 gate_up（act16 保持 zeros），只算 down
        out = k_dn(act16, e_d, idx_i32)
        return out
    def split_moe_decode(x, gate_weight, e_gu, e_d, top_k, n_experts, shared_gu=None, shared_d=None):
        N, Hh = x.shape; K = top_k; tl_dt = KM._TORCH_TO_TL[x.dtype]
        gkey = (N, Hh, n_experts, K, x.dtype)
        if gkey not in KM._kernel_cache:
            KM._kernel_cache[gkey] = (KM.gate_gemv_kernel(N,Hh,n_experts,tl_dt), KM.softmax_topk_kernel(N,n_experts,K,tl_dt))
        k_gv, k_st = KM._kernel_cache[gkey]
        logits = k_gv(x, gate_weight); topk_idx, topk_weight = k_st(logits)
        out = routed_split(x, e_gu, e_d, topk_idx, topk_weight)
        if shared_gu is not None:
            S_INTER = shared_d.shape[0]; skey=(N,Hh,S_INTER,x.dtype)
            if skey not in KM._kernel_cache:
                KM._kernel_cache[skey] = (KM.shared_gate_up_kernel(N,Hh,S_INTER,tl_dt), KM.shared_down_kernel(N,Hh,S_INTER,tl_dt))
            k_sgu, k_sdn = KM._kernel_cache[skey]
            x16 = torch.zeros(N,16,Hh,dtype=x.dtype,device="cuda"); x16[:,0,:]=x
            sact = k_sgu(x16, shared_gu); k_sdn(sact, shared_d, out)
        return out
    adp.moe_decode = split_moe_decode; KM.moe_decode = split_moe_decode

if TAG == "orig_tile":
    # 强制 BLOCK=64
    _od = KM.moe_down_kernel; _osgu = KM.shared_gate_up_kernel; _osdn = KM.shared_down_kernel
    KM.moe_down_kernel = lambda N,H,INTER,E,K,d: _od(N,H,INTER,E,K,d,BLOCK_H=64)
    KM.shared_gate_up_kernel = lambda N,H,S,d: _osgu(N,H,S,d,BLOCK_I=64)
    KM.shared_down_kernel = lambda N,H,S,d: _osdn(N,H,S,d,BLOCK_H=64)
    # 清 cache（之前可能 cache 了 32 的）
    KM._kernel_cache.clear()

# fine_tile = baseline（当前代码已是 32），不额外 patch

if TAG == "persist":
    os.environ["USE_PERSISTENT_MOE"] = "1"

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
