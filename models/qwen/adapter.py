"""
QwenAdapter - Qwen-1 / Qwen2 / Qwen2.5 (GQA + SwiGLU) 适配器。

把 micro-vllm 原本硬编码的 Qwen 逻辑 1:1 包装成 ModelAdapter 钩子，
保证 Qwen 路径行为零回归。
"""
import torch
import torch.nn.functional as F

from models.base import ModelAdapter
from kernel.rmsnorm import rmsnorm_, rmsnorm_residual_gemm as rmsnorm_residual
from kernel.dense_mlp import dense_swiglu

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


class QwenAdapter(ModelAdapter):
    model_type = "qwen"

    # -------------------- 权重预处理 --------------------
    def prepare_weights(self, model, world_size, rank):
        # 幂等：已处理则跳过（标志位 w1 is None）
        if model.transformer.h[0].mlp.w1 is None:
            return
        cfg = model.config
        global_num_heads = cfg.num_attention_heads
        global_kv_heads = getattr(cfg, "num_key_value_heads", global_num_heads)
        head_size = getattr(cfg, "head_dim", cfg.hidden_size // global_num_heads)
        q_dim = global_num_heads * head_size
        kv_dim = global_kv_heads * head_size

        for block in model.transformer.h:
            # MLP: w1(wgate) + w2(wup) → _gu (fused gate_up), c_proj → _d
            w1, w2 = block.mlp.w1.weight, block.mlp.w2.weight
            block.mlp._gu = torch.cat([
                w1.chunk(world_size, dim=0)[rank],
                w2.chunk(world_size, dim=0)[rank]
            ], dim=0).t().contiguous()
            block.mlp._d = block.mlp.c_proj.weight.chunk(world_size, dim=1)[rank].t().contiguous()

            # Attn O
            block.attn._o = block.attn.c_proj.weight.chunk(world_size, dim=1)[rank].t().contiguous()

            # QKV (fused c_attn)
            w_qkv, b_qkv = block.attn.c_attn.weight, block.attn.c_attn.bias
            w_q, w_k, w_v = w_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
            local_qkv = [w.chunk(world_size, dim=0)[rank] for w in (w_q, w_k, w_v)]
            block.attn._qkv_w = torch.cat(local_qkv, dim=0).t().contiguous()
            block.attn._qkv_b = None
            if b_qkv is not None:
                b_q, b_k, b_v = b_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
                local_b = [b.chunk(world_size, dim=0)[rank] for b in (b_q, b_k, b_v)]
                block.attn._qkv_b = torch.cat(local_b, dim=0)

            # 释放原始权重
            block.mlp.w1 = None
            block.mlp.w2 = None
            block.mlp.c_proj = None
            block.attn.c_attn = None
            block.attn.c_proj = None
            torch.cuda.empty_cache()

    # -------------------- 模块访问 --------------------
    def embed(self, model):
        return model.transformer.wte

    def blocks(self, model):
        return model.transformer.h

    def final_norm(self, model):
        return model.transformer.ln_f

    def lm_head(self, model):
        return model.lm_head

    # -------------------- decode 单层钩子 --------------------
    def compute_qkv(self, block, h, graph, bs):
        rmsnorm_(h, block.ln_1.weight, graph._h_buf[:bs], block.ln_1.eps)
        qkv_buf = graph._qkv[:bs]
        torch.matmul(graph._h_buf[:bs], block.attn._qkv_w, out=qkv_buf)
        if block.attn._qkv_b is not None:
            qkv_buf.add_(block.attn._qkv_b)
        return qkv_buf

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        rmsnorm_residual(
            mlp_out_prev, res_prev, block_next.ln_1.weight,
            graph._h_buf[:bs], graph._residual[:bs], block_next.ln_1.eps
        )
        qkv_buf = graph._qkv[:bs]
        torch.matmul(graph._h_buf[:bs], block_next.attn._qkv_w, out=qkv_buf)
        if block_next.attn._qkv_b is not None:
            qkv_buf.add_(block_next.attn._qkv_b)
        return qkv_buf, graph._residual[:bs]

    def attention(self, qkv, block, layer_idx, bs, graph, cache_manager, block_table):
        q, k, v = qkv.reshape(bs, 3, graph.num_heads, graph.head_size).unbind(dim=1)
        k_cache, v_cache = cache_manager.get(layer_idx)
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]

        attn = flash_attn_with_kvcache(
            q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
            k=k.unsqueeze(1), v=v.unsqueeze(1),
            rotary_cos=graph.attention._cos_pool, rotary_sin=graph.attention._sin_pool,
            cache_seqlens=cache_lens, block_table=block_table,
            causal=True, window_size=(-1, -1), rotary_interleaved=False,
            alibi_slopes=None,
            num_splits=0 if bs >= 32 else max(1, 32 // max(1, bs * 4))
        ).squeeze(1)

        out_buf = graph._attn_out[:bs]
        torch.matmul(attn.reshape(bs, -1), block.attn._o, out=out_buf)
        return out_buf

    def compute_ffn(self, block, attn_out, residual, graph, bs):
        rmsnorm_residual(
            attn_out, residual, block.ln_2.weight,
            graph._h_buf[:bs], graph._residual[:bs], block.ln_2.eps
        )
        mlp_out = dense_swiglu(graph._h_buf[:bs], block.mlp._gu, block.mlp._d)
        return mlp_out, graph._residual[:bs]

    # -------------------- prefill 单层钩子 --------------------
    def prefill(self, block, h, layer_idx, B, S, graph, cache_manager, block_table):
        from kernel.rmsnorm import rmsnorm, rmsnorm_residual_fused

        w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
        w_o = block.attn._o

        normed = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)
        qkv = torch.matmul(normed, w_qkv)
        if b_qkv is not None:
            qkv += b_qkv

        q, k, v = qkv.view(B, S, 3, graph.num_heads, graph.head_size).unbind(dim=2)
        q, k = graph.rope.forward(q, k, graph.attention._cos_pool, graph.attention._sin_pool)

        k_cache, v_cache = cache_manager.get(layer_idx)
        # 注意：cache_seqlens 在 prefill 入口由 runner 一次性置 0，逐层写入；这里只读不重置。
        cache_lens = cache_manager._cache_seqlens_buffer[:B]
        attn = flash_attn_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
            cache_seqlens=cache_lens, block_table=block_table, causal=True
        )

        out = torch.matmul(attn.view(B, S, -1), w_o)
        normed, residual = rmsnorm_residual_fused(out, h, block.ln_2.weight, block.ln_2.eps)
        mlp_out = dense_swiglu(normed, block.mlp._gu, block.mlp._d)
        return mlp_out + residual

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        _b = model.transformer.h[0]
        qkv_dim = _b.attn._qkv_w.shape[1]
        o_dim = _b.attn._o.shape[1]
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, qkv_dim, dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, o_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
        }
