import torch
import logging
from typing import List
from kernel.rmsnorm_add import rmsnorm_residual_fused, rmsnorm
from core.parallel_config import all_reduce
from .model_graph import ModelGraphRunner

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelPrefillRunner(ModelGraphRunner):
    def __init__(self, *args, **kwargs):
        # 调用父类初始化，但我们可以选择性地跳过一些 Decode 特有的分配（可选优化）
        super().__init__(*args, **kwargs)
        # Prefill 不需要 CUDA Graph Buffer，可以在这里释放以节省显存（如果需要极致优化）
        # del self._input_ids, self._logits, self._normed ...

    def forward(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        """
        定长 Prefill 前向传播 (batch, seq_len)
        注意：这里不调用 self.decode()，完全重写逻辑
        """
        B, S = input_ids.shape
        h = self.model.transformer.wte(input_ids)

        # 初始化 cache 状态
        cache_lens = cache_manager._cache_seqlens_buffer[:batch_size].zero_()
        block_table = cache_manager._block_table_buffer[:batch_size]

        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
            w_o = block.attn._o

            # Attention 前向
            normed = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None: qkv += b_qkv

            q, k, v = qkv.view(B, S, 3, self.num_heads, self.head_size).unbind(dim=2)
            q, k = self.rope.forward(q, k, self.attention._cos_pool, self.attention._sin_pool)

            k_cache, v_cache = cache_manager.get(layer_idx)
            attn = flash_attn_with_kvcache(
                q=q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                cache_seqlens=cache_lens, block_table=block_table, causal=True
            )

            out = all_reduce(torch.matmul(attn.view(B, S, -1), w_o))
            normed, residual = rmsnorm_residual_fused(out, h, block.ln_2.weight, block.ln_2.eps)

            # MLP 前向 (复用父类编译好的函数)
            mlp_out = self._compiled_mlp(normed, block.mlp._gu, block.mlp._d)
            h = all_reduce(mlp_out).add_(residual)

        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)