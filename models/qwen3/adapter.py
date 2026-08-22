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
import os
import torch

from models.base import ModelAdapter
from kernel.rmsnorm import (
    rmsnorm_, rmsnorm, rmsnorm_residual_gemm as rmsnorm_residual, rmsnorm_residual_fused,
    qk_norm_inplace,
)
from kernel.dense_mlp import dense_swiglu
from kernel.qwen3_decode_attn import qwen3_decode_attn

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


class Qwen3Adapter(ModelAdapter):
    model_type = "qwen3"

    # -------------------- 元信息 --------------------
    def cache_dims(self, cfg):
        num_heads = cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_size = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
        return num_heads, kv_heads, head_size

    def intermediate_size(self, cfg, world_size):
        return cfg.intermediate_size // world_size

    def softmax_scale(self, cfg):
        # QK-Norm 后点积仍按 head_dim scale
        return self.cache_dims(cfg)[2] ** -0.5

    def rope_theta(self, cfg) -> float:
        return getattr(cfg, "rope_theta", None) or 10000.0

    def supports_chunked_prefill(self, cfg) -> bool:
        # Qwen3 prefill_layer 用 flash_attn_with_kvcache(cache_seqlens=position_offsets)，
        # 第 N chunk 的 attention 能读到 cache 中前 N-1 chunk 的 KV；RoPE 按 per-seq offset
        # 从 cos/sin pool gather 正确位置。已验证 chunked vs 非 chunked 输出完全一致。
        return True

    # -------------------- 权重预处理 --------------------
    def prepare_weights(self, model, world_size, rank):
        first = self.blocks(model)[0]
        if getattr(first.self_attn, "_prepared", False):
            return
        cfg = model.config
        num_heads = cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_size = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
        q_dim = num_heads * head_size       # 2048
        kv_dim = kv_heads * head_size        # 1024

        for block in self.blocks(model):
            attn = block.self_attn
            mlp = block.mlp
            # Q/K/V：HF Qwen3 分立 Linear [out, in]，单卡 world_size=1 → .t() 成 [in, out]
            w_q = attn.q_proj.weight.data.chunk(world_size, dim=0)[rank].t().contiguous()  # [hidden, q_dim]
            w_k = attn.k_proj.weight.data.chunk(world_size, dim=0)[rank].t().contiguous()  # [hidden, kv_dim]
            w_v = attn.v_proj.weight.data.chunk(world_size, dim=0)[rank].t().contiguous()  # [hidden, kv_dim]
            attn._qkv_w = torch.cat([w_q, w_k, w_v], dim=1).contiguous()  # [hidden, q_dim+2*kv_dim]
            attn._qkv_b = None  # attention_bias=false
            # O 投影
            attn._o_w = attn.o_proj.weight.data.chunk(world_size, dim=1)[rank].t().contiguous()  # [q_dim, hidden]
            # QK-Norm 权重（RMSNorm on head_dim，shape [head_dim]）
            attn._q_norm_w = attn.q_norm.weight.data.clone()
            attn._k_norm_w = attn.k_norm.weight.data.clone()
            attn._q_norm_eps = getattr(attn.q_norm, "eps", None) or getattr(attn.q_norm, "variance_epsilon", cfg.rms_norm_eps)
            attn._k_norm_eps = getattr(attn.k_norm, "eps", None) or getattr(attn.k_norm, "variance_epsilon", cfg.rms_norm_eps)

            # MLP: dense_swiglu 约定 gu_w = cat([up, gate]).t()（前半 up、后半 gate）。
            # Qwen3 HF: gate_proj=gate, up_proj=up → cat([up_proj, gate_proj]) 对齐。
            w_up = mlp.up_proj.weight.data.chunk(world_size, dim=0)[rank]
            w_gate = mlp.gate_proj.weight.data.chunk(world_size, dim=0)[rank]
            mlp._gu = torch.cat([w_up, w_gate], dim=0).t().contiguous()  # [hidden, 2*inter]
            mlp._d = mlp.down_proj.weight.data.chunk(world_size, dim=1)[rank].t().contiguous()  # [hidden, inter]

            # RMSNorm 权重 + eps
            block._in_ln_w = block.input_layernorm.weight.data.clone()
            block._in_ln_eps = getattr(block.input_layernorm, "eps", None) or \
                getattr(block.input_layernorm, "variance_epsilon", cfg.rms_norm_eps)
            block._post_ln_w = block.post_attention_layernorm.weight.data.clone()
            block._post_ln_eps = getattr(block.post_attention_layernorm, "eps", None) or \
                getattr(block.post_attention_layernorm, "variance_epsilon", cfg.rms_norm_eps)

            # 释放原始权重
            attn.q_proj = attn.k_proj = attn.v_proj = attn.o_proj = None
            attn.q_norm = attn.k_norm = None
            mlp.gate_proj = mlp.up_proj = mlp.down_proj = None
            attn._prepared = True
        torch.cuda.empty_cache()

    # -------------------- 模块访问 --------------------
    def embed(self, model):
        return model.model.embed_tokens

    def blocks(self, model):
        return model.model.layers

    def final_norm(self, model):
        return model.model.norm

    def lm_head(self, model):
        # tie_word_embeddings: lm_head.weight is embed_tokens.weight
        return model.lm_head

    # -------------------- QK-Norm 辅助 --------------------
    @staticmethod
    def _qk_norm_decode(x, norm_w, eps, num_heads, head_size):
        """decode: 对 [bs, num_heads*head_size] 做 per-head RMSNorm（作用在 head_dim 维）。
        等价 transformers: q_proj(h).view(bs, H, hd) → q_norm → 还原。graph 友好（固定形状）。"""
        bs = x.shape[0]
        x = x.view(bs, num_heads, head_size)               # [bs, H, hd]
        var = x.float().pow(2).mean(dim=-1, keepdim=True)  # [bs, H, 1]
        x = (x * torch.rsqrt(var + eps) * norm_w.to(x.dtype)).to(x.dtype)
        return x.view(bs, num_heads * head_size)

    # -------------------- decode 单层钩子 --------------------
    def compute_qkv(self, block, h, graph, bs):
        rmsnorm_(h, block._in_ln_w, graph._h_buf[:bs], block._in_ln_eps)
        qkv_buf = graph._qkv[:bs]
        torch.matmul(graph._h_buf[:bs], block.self_attn._qkv_w, out=qkv_buf)
        self._apply_qk_norm(qkv_buf, block.self_attn, graph, bs)
        return qkv_buf

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        rmsnorm_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block_next._in_ln_eps
        )
        qkv_buf = graph._qkv[:bs]
        torch.matmul(graph._h_buf[:bs], block_next.self_attn._qkv_w, out=qkv_buf)
        self._apply_qk_norm(qkv_buf, block_next.self_attn, graph, bs)
        return qkv_buf, graph._residual[:bs]

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

        # TileLang 轻量 decode attention（bs=1）：单 kernel Q旋转+QK+softmax+PV+存新K/V。
        # 图下实测比 flash 慢（GQA 每 q-head 一 block 致 KV 双读 + 串行 reduce），
        # 暂用 flash；kernel 代码保留待优化（合并 2 q-head/块 + warp reduce 后再启用）。
        if bs == 1 and os.environ.get("MICRO_TILELANG_ATTN"):
            attn_pre = graph._attn_pre[:bs]
            qwen3_decode_attn(
                qkv, k_cache, v_cache, block_table, cache_lens,
                graph.attention._cos_pool, graph.attention._sin_pool,
                graph.num_heads, graph.kv_num_heads, graph.head_size, out=attn_pre)
            torch.matmul(attn_pre, block.self_attn._o_w, out=out_buf)
            return out_buf

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

        torch.matmul(attn.reshape(bs, -1), block.self_attn._o_w, out=out_buf)
        return out_buf

    def compute_ffn(self, block, attn_out, residual, graph, bs):
        rmsnorm_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        mlp_out = dense_swiglu(graph._h_buf[:bs], block.mlp._gu, block.mlp._d)
        return mlp_out, graph._residual[:bs]

    # -------------------- prefill 单层钩子 --------------------
    def prefill_layer(self, block, h, layer_idx, B, S, graph, cache_manager, block_table):
        attn = block.self_attn
        normed = rmsnorm(h, block._in_ln_w, block._in_ln_eps)

        qkv = torch.matmul(normed, attn._qkv_w)            # [B, S, q_dim+2*kv_dim]
        q_dim = graph.num_heads * graph.head_size
        kv_dim = graph.kv_num_heads * graph.head_size
        q = qkv[..., :q_dim].reshape(B, S, graph.num_heads, graph.head_size).contiguous()
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(B, S, graph.kv_num_heads, graph.head_size).contiguous()
        v = qkv[..., q_dim + kv_dim:].reshape(B, S, graph.kv_num_heads, graph.head_size).contiguous()
        # QK-Norm on head_dim 维（rmsnorm 把最后一维 head_dim 当 hidden 归一，逐 head 独立）
        q = rmsnorm(q, attn._q_norm_w, attn._q_norm_eps)
        k = rmsnorm(k, attn._k_norm_w, attn._k_norm_eps)

        k_cache, v_cache = cache_manager.get(layer_idx)
        cache_lens = cache_manager._cache_seqlens_buffer[:B]

        # RoPE (half-split)。chunked prefill 时 cache_lens 非 0 = 该 chunk 在原 prompt
        # 的起始位置；按 per-seq offset 从 cos/sin pool gather 正确行，保证长 prompt
        # 分块续写时位置连续。全 0 走快路径（取 cos[:S]，原一次性 prefill 行为）。
        cos_pool = graph.attention._cos_pool
        sin_pool = graph.attention._sin_pool
        if bool(cache_lens.any()):
            # per-seq offset：pos[b,t] = cache_lens[b] + t
            pos = cache_lens.to(torch.long).unsqueeze(1) + torch.arange(S, device=cache_lens.device).unsqueeze(0)  # [B,S]
            cos = cos_pool[pos]   # [B, S, dim//2]
            sin = sin_pool[pos]
            cos = cos[:, :, None, :]   # [B,S,1,dim//2]
            sin = sin[:, :, None, :]
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        else:
            q, k = graph.rope.forward(q, k, cos_pool, sin_pool)

        attn_out = flash_attn_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
            cache_seqlens=cache_lens, block_table=block_table, causal=True
        )

        out = torch.matmul(attn_out.view(B, S, -1), attn._o_w)
        normed, residual = rmsnorm_residual_fused(out, h, block._post_ln_w, block._post_ln_eps)
        mlp_out = dense_swiglu(normed, block.mlp._gu, block.mlp._d)
        return mlp_out + residual

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        _b = self.blocks(model)[0]
        qkv_dim = _b.self_attn._qkv_w.shape[1]
        o_dim = _b.self_attn._o_w.shape[1]
        # q_dim = num_heads*head_size（o_proj 输入维 = _o_w.shape[0]）
        q_dim = _b.self_attn._o_w.shape[0]
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, qkv_dim, dtype=dtype, device=device),
            # TileLang decode attn 输出（o_proj 前，= num_heads*head_size=2048）；flash 路径不用
            "_attn_pre": torch.empty(max_bs, q_dim, dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, o_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
        }
