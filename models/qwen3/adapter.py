"""Qwen3Adapter - Qwen3 (GQA + SwiGLU + QK-Norm) 适配器。

与老 Qwen-1 (model_type=qwen, MHA, 旧命名) 的关键差异：
- HF 原生命名：model.layers[i].{self_attn, mlp, input_layernorm, post_attention_layernorm}，
  self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}，mlp.{gate_proj,up_proj,down_proj}，
  model.{embed_tokens,layers,norm}，lm_head（tie_word_embeddings=true 时 = embed_tokens）。
- GQA：num_attention_heads=16, num_key_value_heads=8（q/kv 头数不等，不能像 Qwen-1 那样
  用 qkv.reshape(bs,3,num_heads,head_size) 三等分；必须按 [q_dim,kv_dim,kv_dim] split）。
- head_dim 独立配置（cfg.head_dim=128，非 hidden/heads=64）。
- QK-Norm：q_proj/k_proj 后、RoPE 前对 q/k 各做一次 RMSNorm（作用在 head_dim 维，
  q_norm/k_norm.weight shape [head_dim]）。decode 路径在进 flash_attn 前对 q/k 做 norm，
  RoPE 交给 flash_attn_with_kvcache 内部按 cache_seqlens 旋转；prefill 路径显式 norm 后
  调 graph.rope.forward 旋转。
- RoPE：标准 Llama 风格 rotate_half（half-split），rotary_interleaved=False，rope_theta=1e6。
  复用 PagedAttention 的 _cos_pool/_sin_pool（已按 adapter.rope_theta 用 1e6 预计算）。
- tie_word_embeddings=true：lm_head.weight is embed_tokens.weight。
"""
import torch

from models.base import ModelAdapter
from kernel.rmsnorm import (
    rmsnorm_, rmsnorm, rmsnorm_residual_gemm as rmsnorm_residual, rmsnorm_residual_fused,
    qk_norm_inplace,
)
from kernel.dense_mlp import dense_swiglu
from kernel.gemv import gemv_or_matmul
from kernel.rotary import qk_norm_rope_inplace

try:
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
except ImportError:
    flash_attn_with_kvcache = None
    flash_attn_varlen_func = None

from core.cache_manager import store_kvcache


class Qwen3Adapter(ModelAdapter):
    model_type = "qwen3"

    # decode attention 路径：True=prerope+store+pure-flash（对齐 nano，省 50us/层）；
    # False=flash internal-rotary+k=/v=（旧逻辑 fallback）。
    use_prerope_decode = True

    # -------------------- 元信息 --------------------
    def supports_chunked_prefill(self, cfg) -> bool:
        # Qwen3 prefill_layer 用 flash_attn_varlen_func(block_table=...)，第 N chunk 的
        # attention 经 block_table 读 cache 中前 N-1 chunk 的 KV；RoPE 按 per-token position
        # 从 cos/sin pool gather 正确位置。已验证 chunked vs 非 chunked 输出完全一致。
        return True

    def supports_varlen_prefill(self, cfg) -> bool:
        # flash_attn_varlen_func 支持 block_table 读 paged cache，同 batch 内各 seq 长度
        # 可不同（cu_seqlens 掩码），无需等长分组/padding。
        return True

    @staticmethod
    def _ln_eps(ln, cfg):
        """兼容 HF RMSNorm 的两种 eps 属性名（eps / variance_epsilon），回退 cfg.rms_norm_eps。"""
        return getattr(ln, "eps", None) or getattr(ln, "variance_epsilon", cfg.rms_norm_eps)

    # -------------------- 权重预处理 --------------------
    def prepare_weights(self, model, world_size, rank):
        first = self.blocks(model)[0]
        if getattr(first.self_attn, "_prepared", False):
            return
        cfg = model.config

        for block in self.blocks(model):
            attn = block.self_attn
            mlp = block.mlp
            # 权重统一存 [N,K]=[out,in]（HF 原始布局，不 .t()）：GEMV 友好（每输出行连续 K），
            # 手写 gemv_v2 kernel 直接读；torch.matmul 路径用 W.t()（opT，裸 cuBLAS 无损）。
            # Q/K/V：HF [out,in]，cat 沿输出维 dim=0 → [q_dim+2*kv_dim, hidden]
            w_q = attn.q_proj.weight.data.chunk(world_size, dim=0)[rank]  # [q_dim, hidden]
            w_k = attn.k_proj.weight.data.chunk(world_size, dim=0)[rank]  # [kv_dim, hidden]
            w_v = attn.v_proj.weight.data.chunk(world_size, dim=0)[rank]  # [kv_dim, hidden]
            attn._qkv_w = torch.cat([w_q, w_k, w_v], dim=0).contiguous()  # [q_dim+2*kv_dim, hidden]
            attn._qkv_b = None  # attention_bias=false
            # O 投影：[hidden, q_dim]（HF o_proj.weight = [out=hidden, in=q_dim]）
            attn._o_w = attn.o_proj.weight.data.chunk(world_size, dim=1)[rank].contiguous()  # [hidden, q_dim]
            # QK-Norm 权重（RMSNorm on head_dim，shape [head_dim]）
            attn._q_norm_w = attn.q_norm.weight.data.clone()
            attn._k_norm_w = attn.k_norm.weight.data.clone()
            attn._q_norm_eps = self._ln_eps(attn.q_norm, cfg)
            attn._k_norm_eps = self._ln_eps(attn.k_norm, cfg)

            # MLP: dense_swiglu 约定 gu_w 输出维 [up|gate] 顺序（前半 up、后半 gate）。
            # [N,K] 布局：cat 沿输出维 dim=0 → [2*inter, hidden]，前半 up、后半 gate。
            w_up = mlp.up_proj.weight.data.chunk(world_size, dim=0)[rank]    # [inter, hidden]
            w_gate = mlp.gate_proj.weight.data.chunk(world_size, dim=0)[rank]  # [inter, hidden]
            mlp._gu = torch.cat([w_up, w_gate], dim=0).contiguous()  # [2*inter, hidden]
            mlp._d = mlp.down_proj.weight.data.chunk(world_size, dim=1)[rank].contiguous()  # [hidden, inter]

            # RMSNorm 权重 + eps
            block._in_ln_w = block.input_layernorm.weight.data.clone()
            block._in_ln_eps = self._ln_eps(block.input_layernorm, cfg)
            block._post_ln_w = block.post_attention_layernorm.weight.data.clone()
            block._post_ln_eps = self._ln_eps(block.post_attention_layernorm, cfg)

            # 释放原始权重
            attn.q_proj = attn.k_proj = attn.v_proj = attn.o_proj = None
            attn.q_norm = attn.k_norm = None
            mlp.gate_proj = mlp.up_proj = mlp.down_proj = None
            attn._prepared = True
        torch.cuda.empty_cache()

    # -------------------- decode 单层钩子 --------------------
    def _project_qkv(self, attn, graph, bs):
        """normed h（graph._h_buf[:bs]）→ QKV 投影，写 graph._qkv[:bs]。
        QK-Norm：prerope 路径在 attention 里与 RoPE 融合（qk_norm_rope_inplace），
        internal-rotary 路径在此处单独做（_apply_qk_norm）。"""
        qkv_buf = graph._qkv[:bs]
        gemv_or_matmul(graph._h_buf[:bs], attn._qkv_w, qkv_buf, "MICRO_GEMV_QKV")
        if not getattr(self, "use_prerope_decode", False):
            self._apply_qk_norm(qkv_buf, attn, graph, bs)
        return qkv_buf

    def compute_qkv(self, block, h, graph, bs):
        rmsnorm_(h, block._in_ln_w, graph._h_buf[:bs], block._in_ln_eps)
        return self._project_qkv(block.self_attn, graph, bs)

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        rmsnorm_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block_next._in_ln_eps
        )
        return self._project_qkv(block_next.self_attn, graph, bs), graph._residual[:bs]

    def _apply_qk_norm(self, qkv_buf, attn, graph, bs):
        """对融合 qkv buffer 的 q 段、k 段原地做 QK-Norm（per-head RMSNorm on head_dim）。

        单个 Triton kernel 直接在 qkv_buf 上原地 norm（每个 program 处理一个 head，
        两遍顺序执行原地安全），替代旧 PyTorch 原生 op 的碎片 kernel
        （cast/pow/mean/rsqrt/mul 共 ~6 个 elementwise+reduce kernel/层/head）。
        """
        q_dim = graph.num_heads * graph.head_size
        kv_dim = graph.kv_num_heads * graph.head_size
        qk_norm_inplace(qkv_buf, bs, q_dim, kv_dim,
                        attn._q_norm_w, attn._k_norm_w,
                        graph.num_heads, graph.kv_num_heads, graph.head_size,
                        attn._q_norm_eps)

    def attention(self, qkv, block, layer_idx, bs, graph, cache_manager, block_table):
        q_dim = graph.num_heads * graph.head_size
        kv_dim = graph.kv_num_heads * graph.head_size
        q = qkv[:, :q_dim].view(bs, graph.num_heads, graph.head_size)
        k = qkv[:, q_dim:q_dim + kv_dim].view(bs, graph.kv_num_heads, graph.head_size)
        v = qkv[:, q_dim + kv_dim:].view(bs, graph.kv_num_heads, graph.head_size)

        k_cache, v_cache = cache_manager.get(layer_idx)
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]
        out_buf = graph._attn_out[:bs]

        # prerope+store 路径（对齐 nano-vllm，省 50us/层）：flash 前显式旋转 q/k、
        # store k/v，flash 跑纯 attention（无 rotary_cos/sin、无 k=/v=）。
        # internal-rotary 路径保留为 fallback（旧逻辑）。
        if getattr(self, "use_prerope_decode", False):
            # QK-Norm + RoPE 融合（prerope 路径）：对 q 段、k 段原地 norm+rotate，
            # 替代分离的 _apply_qk_norm + apply_rope_decode（省中间读+写）。
            sa = block.self_attn
            qk_norm_rope_inplace(qkv, bs, 0, graph.num_heads, graph.head_size,
                                 sa._q_norm_w, graph.attention._cos_pool,
                                 graph.attention._sin_pool, cache_lens, sa._q_norm_eps)
            qk_norm_rope_inplace(qkv, bs, q_dim, graph.kv_num_heads, graph.head_size,
                                 sa._k_norm_w, graph.attention._cos_pool,
                                 graph.attention._sin_pool, cache_lens, sa._k_norm_eps)
            # q/k 视图已在上面建好（共享 qkv 存储），融合 kernel 原地改了 q/k，视图随之更新
            store_kvcache(k, v, k_cache, v_cache, graph._slot_mapping[:bs])
            # flash 读 cache_seqlens+1（含刚 store 的当前 token）
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=graph._flash_seqlens[:bs], block_table=block_table,
                causal=True, window_size=(-1, -1), alibi_slopes=None,
                num_splits=1 if bs == 1 else (0 if bs >= 32 else max(1, 32 // max(1, bs * 4)))
            ).squeeze(1)
        else:
            # flash_attn GQA：q 头数(16) 可被 kv 头数(8) 整除，连续分组（head 0,1→kv0）。
            # RoPE 由 flash_attn 按 cache_seqlens 内部旋转（rotary_cos/sin half-split, interleaved=False）。
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
                k=k.unsqueeze(1), v=v.unsqueeze(1),
                rotary_cos=graph.attention._cos_pool, rotary_sin=graph.attention._sin_pool,
                cache_seqlens=cache_lens, block_table=block_table,
                causal=True, window_size=(-1, -1), rotary_interleaved=False,
                alibi_slopes=None,
                # num_splits：bs=1 短 KV 无需 split（split=1 省 split+combine 两轮 kernel）；
                # 小 batch 按 32//bs*4 给 splits 充分并行；大 batch(≥32) 让 flash 自动选(0)。
                num_splits=1 if bs == 1 else (0 if bs >= 32 else max(1, 32 // max(1, bs * 4)))
            ).squeeze(1)

        return gemv_or_matmul(attn.reshape(bs, -1), block.self_attn._o_w, out_buf, "MICRO_GEMV_O")

    def compute_ffn(self, block, attn_out, residual, graph, bs):
        rmsnorm_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        # dense_swiglu：Qwen3 权重 [N,K] 布局（w_is_nk=True），M=1 decode 走 gemv_v2
        mlp_out = dense_swiglu(graph._h_buf[:bs], block.mlp._gu, block.mlp._d, bs, w_is_nk=True)
        return mlp_out, graph._residual[:bs]

    # -------------------- prefill 单层钩子（变长：h=[total_tokens, hidden]）--------------------
    def prefill_layer(self, block, h, layer_idx, graph, cache_manager, meta):
        attn = block.self_attn
        normed = rmsnorm(h, block._in_ln_w, block._in_ln_eps)

        qkv = torch.matmul(normed, attn._qkv_w.t())            # [total, q_dim+2*kv_dim]（W=[N,K]）
        q_dim = graph.num_heads * graph.head_size
        kv_dim = graph.kv_num_heads * graph.head_size
        q = qkv[..., :q_dim].reshape(-1, graph.num_heads, graph.head_size).contiguous()
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(-1, graph.kv_num_heads, graph.head_size).contiguous()
        v = qkv[..., q_dim + kv_dim:].reshape(-1, graph.kv_num_heads, graph.head_size).contiguous()
        # QK-Norm on head_dim 维（rmsnorm 把最后一维 head_dim 当 hidden 归一，逐 head 独立）
        q = rmsnorm(q, attn._q_norm_w, attn._q_norm_eps)
        k = rmsnorm(k, attn._k_norm_w, attn._k_norm_eps)

        # RoPE (half-split)：按 per-token 绝对位置 position_ids 从 cos/sin pool gather。
        # 变长拼接下每 token 位置由 meta.position_ids 给出（完整 prefill=0..L-1，chunked 续写从 prefill_done 起）。
        cos_pool = graph.attention._cos_pool
        sin_pool = graph.attention._sin_pool
        pos = meta.position_ids.long()                         # [total]
        cos = cos_pool[pos].unsqueeze(1)                       # [total, 1, dim//2]
        sin = sin_pool[pos].unsqueeze(1)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        # 写入 paged cache：本步算出的 k/v 按 slot_mapping scatter，供 varlen attention 经
        # block_table 读回完整 KV（含 chunked 续写时的 cache 前缀）。
        k_cache, v_cache = cache_manager.get(layer_idx)
        store_kvcache(k, v, k_cache, v_cache, meta.slot_mapping)

        # varlen attention：cu_seqlens 掩码各 seq 边界，block_table 读 paged cache。
        attn_out = flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=meta.cu_seqlens_q, cu_seqlens_k=meta.cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q, max_seqlen_k=meta.max_seqlen_k,
            softmax_scale=graph.head_size ** -0.5, causal=True,
            block_table=meta.block_table,
        )

        out = torch.matmul(attn_out.view(-1, graph.num_heads * graph.head_size), attn._o_w.t())
        normed, residual = rmsnorm_residual_fused(out, h, block._post_ln_w, block._post_ln_eps)
        mlp_out = dense_swiglu(normed, block.mlp._gu, block.mlp._d, out.shape[0], w_is_nk=True)
        return mlp_out + residual

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        _b = self.blocks(model)[0]
        # 权重 [N,K]=[out,in] 布局：shape[0]=out(N), shape[1]=in(K=hidden)
        qkv_dim = _b.self_attn._qkv_w.shape[0]   # q_dim+2*kv_dim
        o_dim = _b.self_attn._o_w.shape[0]       # hidden（o_proj 输出维）
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, qkv_dim, dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, o_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
        }
