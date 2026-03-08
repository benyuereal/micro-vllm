import logging
import torch
import torch.nn.functional as F
from typing import Dict, List

from core.paged_attention import PagedAttention
from kernel.swiglu import swiglu_fused as swiglu
from kernel.rmsnorm_add import rmsnorm_residual_fused, rmsnorm
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

        # 本地维度（已TP切分）
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

        # 先准备权重，再分配Buffer（确保维度匹配）
        self.prepare()
        self._allocate_buffers()

    def prepare(self):
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
            w1, w2 = mlp.w1.weight, mlp.w2.weight
            mlp._gu = torch.cat([
                w1.chunk(self.world_size, dim=0)[self.rank],
                w2.chunk(self.world_size, dim=0)[self.rank]
            ], dim=0).t().contiguous()

            # MLP Down (Row Parallel)
            mlp._d = mlp.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # Attention Output (Row Parallel)
            block.attn._o = block.attn.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # QKV (Column Parallel)
            w_qkv, b_qkv = block.attn.c_attn.weight, block.attn.c_attn.bias
            w_q, w_k, w_v = w_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
            local_qkv = [w.chunk(self.world_size, dim=0)[self.rank] for w in (w_q, w_k, w_v)]
            block.attn._qkv_w = torch.cat(local_qkv, dim=0).t().contiguous()

            if b_qkv is not None:
                b_q, b_k, b_v = b_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
                local_b = [b.chunk(self.world_size, dim=0)[self.rank] for b in (b_q, b_k, b_v)]
                block.attn._qkv_b = torch.cat(local_b, dim=0)
            else:
                block.attn._qkv_b = None

        logger.info(f"✅ 权重预缓存完成 (Rank {self.rank})")

    def _allocate_buffers(self):
        """预分配Decode阶段的中间Buffer"""
        max_b = self.max_batch_size
        self._input_ids = torch.empty(max_b, dtype=torch.long, device=self.device)
        self._logits = torch.empty(max_b, self.vocab_size, dtype=self.dtype, device=self.device)

        # 从第一个block获取切分后的实际维度
        _block = self.model.transformer.h[0]
        qkv_dim = _block.attn._qkv_w.shape[1]
        gu_dim = _block.mlp._gu.shape[1]
        o_dim = _block.attn._o.shape[1]

        # 分配中间Buffer
        self._qkv_buffer = torch.empty(max_b, qkv_dim, dtype=self.dtype, device=self.device)
        self._gate_up_buffer = torch.empty(max_b, gu_dim, dtype=self.dtype, device=self.device)
        self._attn_output_buffer = torch.empty(max_b, o_dim, dtype=self.dtype, device=self.device)
        self._mlp_output_buffer = torch.empty(max_b, o_dim, dtype=self.dtype, device=self.device)

    def decode(self, input_ids, batch_size, cache_manager, use_graph_cache):
        """单token前向（decode步）"""
        h = self.model.transformer.wte(input_ids)

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
            w_o = block.attn._o

            # Attention (复用QKV Buffer)
            normed = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)
            qkv = self._qkv_buffer[:batch_size]
            torch.matmul(normed, w_qkv, out=qkv)
            if b_qkv is not None:
                qkv += b_qkv

            q, k, v = qkv.reshape(batch_size, 3, self.num_heads, self.head_size).unbind(dim=1)

            k_cache, v_cache = cache_manager.get(layer_idx)
            cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]
            block_table = (
                cache_manager._block_table_buffer[:batch_size]
                if use_graph_cache else
                torch.zeros(batch_size, self.attention.max_blocks, dtype=torch.int32, device=self.device)
            )

            attn = flash_attn_with_kvcache(
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
                num_splits=max(1, 32 // max(1, batch_size * 4))  # 修复除0
            ).squeeze(1)

            attn_out = self._attn_output_buffer[:batch_size]
            torch.matmul(attn.reshape(batch_size, -1), w_o, out=attn_out)
            out = all_reduce(attn_out)

            normed, residual = rmsnorm_residual_fused(out, h, block.ln_2.weight, block.ln_2.eps)

            gate_up = self._gate_up_buffer[:batch_size]
            torch.matmul(normed, block.mlp._gu, out=gate_up)
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            mlp_out = self._mlp_output_buffer[:batch_size]
            torch.matmul(activated, block.mlp._d, out=mlp_out)
            h = all_reduce(mlp_out).add_(residual)

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

    def prefill(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        """定长Prefill（batch, seq_len）"""
        B, S = input_ids.shape
        h = self.model.transformer.wte(input_ids)

        # 初始化cache状态
        cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size].zero_()
        block_table = cache_manager._block_table_buffer[:batch_size]

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
            w_o = block.attn._o

            normed = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None:
                qkv += b_qkv

            q, k, v = qkv.view(B, S, 3, self.num_heads, self.head_size).unbind(dim=2)

            # 手动RoPE
            q, k = self.rope.forward(q, k, self.attention._cos_pool, self.attention._sin_pool)

            k_cache, v_cache = cache_manager.get(layer_idx)
            attn = flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True
            )

            out = all_reduce(torch.matmul(attn.view(B, S, -1), w_o))

            normed, residual = rmsnorm_residual_fused(out, h, block.ln_2.weight, block.ln_2.eps)

            gate_up = torch.matmul(normed, block.mlp._gu)
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            mlp_out = torch.matmul(activated, block.mlp._d)
            h = all_reduce(mlp_out).add_(residual)

        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)