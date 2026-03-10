import logging
import torch
from typing import Dict, List

from core.paged_attention import PagedAttention
from kernel.swiglu import swiglu_fused as swiglu
from kernel.rmsnorm_add import rmsnorm_residual_fused, rmsnorm, rmsnorm_
from kernel.rmsnorm_residual import rmsnorm_residual_

from .rope import RoPE
from core.parallel_config import get_rank, get_world_size, all_reduce

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelGraphRunner:
    def __init__(self, model, num_layers: int, num_heads: int, head_size: int,
                 kv_num_heads: int, hidden_dim: int, intermediate_size: int,
                 device: str, max_batch_size: int = 32, dtype: torch.dtype = torch.bfloat16,
                 top_k: int = 1000):
        self.model = model
        self.num_layers = num_layers
        self.rank, self.world_size = get_rank(), get_world_size()

        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.intermediate_size = intermediate_size
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.vocab_size = model.config.vocab_size
        self.device = device
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.top_k = top_k

        self.attention = PagedAttention(num_heads, head_size, kv_num_heads, device, max_batch_size)
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._ready = False
        self.rope = RoPE()

        # 先准备权重，再分配Buffer
        self._prep_weights()
        self._alloc_bufs()

    def _prep_weights(self):
        """预缓存转置权重 & TP切分（简化QKV处理）"""
        cfg = self.model.config
        global_num_heads = cfg.num_attention_heads
        global_kv_heads = getattr(cfg, "num_key_value_heads", global_num_heads)
        head_size = self.head_size
        q_dim = global_num_heads * head_size
        kv_dim = global_kv_heads * head_size

        for block in self.model.transformer.h:
            mlp = block.mlp

            # MLP Gate/Up (Column Parallel)
            w1 = mlp.w1.weight
            w2 = mlp.w2.weight
            local_w1 = w1.chunk(self.world_size, dim=0)[self.rank]
            local_w2 = w2.chunk(self.world_size, dim=0)[self.rank]
            mlp._gu = torch.cat([local_w1, local_w2], dim=0).t().contiguous()

            # MLP Down (Row Parallel)
            mlp._d = mlp.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # Attention Output (Row Parallel)
            block.attn._o = block.attn.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # QKV (Column Parallel)
            w_qkv = block.attn.c_attn.weight
            b_qkv = block.attn.c_attn.bias
            w_q, w_k, w_v = w_qkv.split([q_dim, kv_dim, kv_dim], dim=0)

            local_q = w_q.chunk(self.world_size, dim=0)[self.rank]
            local_k = w_k.chunk(self.world_size, dim=0)[self.rank]
            local_v = w_v.chunk(self.world_size, dim=0)[self.rank]
            block.attn._qkv_w = torch.cat([local_q, local_k, local_v], dim=0).t().contiguous()

            if b_qkv is not None:
                b_q, b_k, b_v = b_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
                local_bq = b_q.chunk(self.world_size, dim=0)[self.rank]
                local_bk = b_k.chunk(self.world_size, dim=0)[self.rank]
                local_bv = b_v.chunk(self.world_size, dim=0)[self.rank]
                block.attn._qkv_b = torch.cat([local_bq, local_bk, local_bv], dim=0)
            else:
                block.attn._qkv_b = None

        logger.info(f"✅ 权重预缓存完成 (Rank {self.rank})")

    def _alloc_bufs(self):
        """预分配Decode阶段的中间Buffer"""
        max_b = self.max_batch_size
        device = self.device
        dtype = self.dtype

        # 基础Buffer
        self._input_ids = torch.empty(max_b, dtype=torch.long, device=device)
        self._logits = torch.empty(max_b, self.vocab_size, dtype=dtype, device=device)

        # 中间计算Buffer（根据维度直接定义）
        # QKV维度 = (num_heads + 2*kv_num_heads) * head_size
        qkv_dim = (self.num_heads + 2 * self.kv_num_heads) * self.head_size
        # GateUp维度 = intermediate_size (已TP切分)
        gu_dim = self.intermediate_size
        # Output维度 = num_heads * head_size (已TP切分)
        o_dim = self.num_heads * self.head_size

        self._normed_buffer = torch.empty((max_b, self.hidden_dim), dtype=dtype, device=device)
        self._qkv_buffer = torch.empty(max_b, qkv_dim, dtype=dtype, device=device)
        self._gate_up = torch.empty(max_b, gu_dim, dtype=dtype, device=device)
        self._attn_output_buffer = torch.empty(max_b, o_dim, dtype=dtype, device=device)
        self._mlp_out = torch.empty(max_b, o_dim, dtype=dtype, device=device)

    def decode(self, input_ids, batch_size, cache_manager, use_graph_cache):
        """单token前向（decode步）"""
        h = self.model.transformer.wte(input_ids)

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            w_qkv = block.attn._qkv_w
            b_qkv = block.attn._qkv_b
            w_o = block.attn._o

            # Attention 计算
            # RMSNorm
            rmsnorm_(h, block.ln_1.weight, self._normed_buffer[:batch_size], block.ln_1.eps)

            # QKV投影
            torch.matmul(self._normed_buffer[:batch_size], w_qkv, out=self._qkv_buffer[:batch_size])
            if b_qkv is not None:
                self._qkv_buffer[:batch_size].add_(b_qkv)

            # QKV拆分
            qkv_reshaped = self._qkv_buffer[:batch_size].reshape(batch_size, 3, self.num_heads, self.head_size)
            q, k, v = qkv_reshaped.unbind(1)

            # 获取Cache
            k_cache, v_cache = cache_manager.get(layer_idx)
            cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]

            # Block Table处理
            if use_graph_cache:
                block_table = cache_manager._block_table_buffer[:batch_size]
            else:
                block_table = torch.zeros(batch_size, self.attention.max_blocks, dtype=torch.int32, device=self.device)

            # FlashAttention计算
            attn_out = flash_attn_with_kvcache(
                q=q.unsqueeze(1),
                k_cache=k_cache,
                v_cache=v_cache,
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                rotary_cos=self.attention._cos_pool,
                rotary_sin=self.attention._sin_pool,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
                num_splits=max(1, 32 // max(1, batch_size * 4))
            )
            attn_out = attn_out.squeeze(1)

            # Attention输出投影
            attn_out_reshaped = attn_out.reshape(batch_size, -1)
            torch.matmul(attn_out_reshaped, w_o, out=self._attn_output_buffer[:batch_size])
            out = all_reduce(self._attn_output_buffer[:batch_size])

            # MLP计算
            # RMSNorm + Residual
            normed, residual = rmsnorm_residual_(out, h, block.ln_2.weight, self._normed_buffer[:batch_size],
                                                 block.ln_2.eps)

            # GateUp投影
            torch.matmul(self._normed_buffer[:batch_size], block.mlp._gu, out=self._gate_up[:batch_size])

            # SwiGLU激活
            activated = swiglu(self._gate_up[:batch_size])

            # MLP输出投影
            torch.matmul(activated, block.mlp._d, out=self._mlp_out[:batch_size])

            # AllReduce + Residual
            h = all_reduce(self._mlp_out[:batch_size])
            h = h.add_(residual)

        # 最后的LayerNorm和LM Head
        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)

    def prefill(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        """定长Prefill（batch, seq_len）"""
        B, S = input_ids.shape
        h = self.model.transformer.wte(input_ids)

        # 初始化cache状态
        cache_manager._cache_seqlens_buffer[:batch_size].zero_()

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            w_qkv = block.attn._qkv_w
            b_qkv = block.attn._qkv_b
            w_o = block.attn._o

            # Attention计算
            # RMSNorm
            normed_x = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)

            # QKV投影
            qkv = normed_x @ w_qkv
            if b_qkv is not None:
                qkv = qkv + b_qkv

            # QKV拆分
            qkv_reshaped = qkv.view(B, S, 3, self.num_heads, self.head_size)
            q, k, v = qkv_reshaped.unbind(2)

            # RoPE编码
            q, k = self.rope.forward(q, k, self.attention._cos_pool, self.attention._sin_pool)

            # 获取Cache
            k_cache, v_cache = cache_manager.get(layer_idx)
            cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]
            block_table = cache_manager._block_table_buffer[:batch_size]

            # FlashAttention计算
            attn_out = flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True
            )

            # Attention输出投影
            attn_out_reshaped = attn_out.view(B, S, -1)
            attn_out_proj = attn_out_reshaped @ w_o
            out = all_reduce(attn_out_proj)

            # MLP计算
            # RMSNorm + Residual
            normed, residual = rmsnorm_residual_fused(out, h, block.ln_2.weight, block.ln_2.eps)

            # GateUp投影
            gate_up = normed @ block.mlp._gu

            # SwiGLU激活
            activated = swiglu(gate_up)

            # MLP输出投影
            mlp_out = activated @ block.mlp._d

            # AllReduce + Residual
            h = all_reduce(mlp_out)
            h = h.add_(residual)

        # 最后的LayerNorm和LM Head
        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)

    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._ready:
            return

        logger.info("🎯 开始捕获 CUDA Graph ...")
        for bs in batch_sizes:
            g = torch.cuda.CUDAGraph()

            # Warmup
            dummy = torch.randint(0, self.vocab_size, (bs,), device=self.device)
            for _ in range(3):
                with torch.no_grad():
                    self.decode(dummy, bs, cache_manager, use_graph_cache=False)
            torch.cuda.synchronize()

            with torch.no_grad(), torch.cuda.graph(g):
                self._logits[:bs] = self.decode(
                    self._input_ids[:bs], bs, cache_manager, use_graph_cache=True
                )
            self._graphs[bs] = g
            logger.info(f"   - Batch size {bs} OK")

        self._ready = True

    def forward(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        self._input_ids[:batch_size] = input_ids

        if batch_size not in self._graphs:
            return self.decode(input_ids, batch_size, cache_manager, use_graph_cache=False)

        self._graphs[batch_size].replay()
        return self._logits[:batch_size]