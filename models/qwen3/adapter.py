"""Qwen3Adapter - Qwen3 (GQA + SwiGLU + QK-Norm) 适配器。

继承 GQAAdapter（models/gqa_base.py）复用公共骨架（_ln_eps / prepare_weights bf16
[N,K] 布局 / compute_qkv / compute_next_qkv / compute_ffn / attention prerope 路径 /
alloc_bufs）。本文件只保留 Qwen3 的差异点：

- 纯 bf16（无 W8A16/Marlin int8 路径）：线性分派用 gemv_or_matmul（基类默认 _lin）。
- 非 1-centered RMSNorm：norm 变体用 rmsnorm*（基类默认 _norm_inplace/_norm_residual）。
- QK-Norm+RoPE 用 qk_norm_rope_inplace（全 head_dim，非 partial）。
- qkv 布局 [q|k|v]（k 紧跟 q，_k_offset=q_dim，基类默认）。
- 无 attn_output_gate（基类默认 no-op）。
- prefill 用 flash_attn_varlen_func + 显式 QK-Norm + half-split RoPE（Qwen3 特有，
  基类不抽 prefill，因 Qwen3.5 prefill 有 GDN 分支差异大）。

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

from models.gqa_base import GQAAdapter
from kernel.rmsnorm import rmsnorm, rmsnorm_residual, qk_norm_inplace
from kernel.dense_mlp import dense_swiglu
from kernel.rotary import qk_norm_rope_inplace

from core.cache_manager import store_kvcache
from core.parallel_config import all_reduce
from kernel.dflash_ops import rope_half_split  # 公共 half-split RoPE（prefill 路径按 per-token position gather 后逐 token 旋转）

try:
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
except ImportError:
    flash_attn_with_kvcache = None
    flash_attn_varlen_func = None


class Qwen3Adapter(GQAAdapter):
    model_type = "qwen3"

    # -------------------- 元信息 --------------------
    def supports_chunked_prefill(self, cfg) -> bool:
        # Qwen3 prefill 用 flash_attn_varlen_func(block_table=...)，第 N chunk 的
        # attention 经 block_table 读 cache 中前 N-1 chunk 的 KV；RoPE 按 per-token position
        # 从 cos/sin pool gather 正确位置。已验证 chunked vs 非 chunked 输出完全一致。
        return True

    # -------------------- 虚方法 override --------------------
    def _qk_norm_rope(self, qkv, bs, seg_offset, num_heads, head_size,
                      norm_w, cos_pool, sin_pool, cache_lens, eps):
        """Qwen3：全 head_dim QK-Norm + half-split RoPE（非 partial）。"""
        qk_norm_rope_inplace(qkv, bs, seg_offset, num_heads, head_size,
                             norm_w, cos_pool, sin_pool, cache_lens, eps)

    def _apply_qk_norm(self, qkv_buf, attn, graph, bs):
        """对融合 qkv buffer 的 q 段、k 段原地做 QK-Norm（per-head RMSNorm on head_dim）。

        单个 Triton kernel 直接在 qkv_buf 上原地 norm（每个 program 处理一个 head，
        两遍顺序执行原地安全），替代旧 PyTorch 原生 op 的碎片 kernel
        （cast/pow/mean/rsqrt/mul 共 ~6 个 elementwise+reduce kernel/层/head）。
        仅 internal-rotary fallback 路径用（use_prerope_decode=False）。
        """
        q_dim = graph.num_heads * graph.head_size
        kv_dim = graph.kv_num_heads * graph.head_size
        qk_norm_inplace(qkv_buf, bs, q_dim, kv_dim,
                        attn._q_norm_w, attn._k_norm_w,
                        graph.num_heads, graph.kv_num_heads, graph.head_size,
                        attn._q_norm_eps)

    def _project_qkv(self, attn, graph, bs):
        """normed h（graph._h_buf[:bs]）→ QKV 投影，写 graph._qkv[:bs]。
        QK-Norm：prerope 路径在 attention 里与 RoPE 融合（qk_norm_rope_inplace），
        internal-rotary 路径在此处单独做（_apply_qk_norm）。"""
        qkv_buf = graph._qkv[:bs]
        self._lin(graph._h_buf[:bs], attn._qkv_w, qkv_buf, "MICRO_GEMV_QKV")
        if not self.use_prerope_decode:
            self._apply_qk_norm(qkv_buf, attn, graph, bs)
        return qkv_buf

    # -------------------- prefill 单层钩子（变长：h=[total_tokens, hidden]）--------------------
    def prefill(self, block, h, layer_idx, graph, cache_manager, meta):
        attn = block.self_attn
        normed = rmsnorm(h, block._in_ln_w, block._in_ln_eps)

        qkv = self._lin_prefill(normed, attn._qkv_w)  # [total, q_dim+2*kv_dim]（W=[N,K]）
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
        q = rope_half_split(q, cos, sin)
        k = rope_half_split(k, cos, sin)

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

        out = self._lin_prefill(attn_out.view(-1, graph.num_heads * graph.head_size), attn._o_w)
        out = all_reduce(out)  # TP: o_proj 按输入维切分，各 rank 持部分和，须 allreduce
        normed, residual = rmsnorm_residual(out, h, block._post_ln_w, block._post_ln_eps)
        mlp_out = dense_swiglu(normed, block.mlp._gu, block.mlp._d, out.shape[0], w_is_nk=True)
        mlp_out = all_reduce(mlp_out)  # TP: down_proj 按输入维切分，各 rank 持部分和，须 allreduce
        return mlp_out + residual
