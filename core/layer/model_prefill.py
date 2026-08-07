import torch
import logging
from typing import List
from kernel.rmsnorm import rmsnorm_residual_fused, rmsnorm
from core.parallel_config import all_reduce
from .model_graph import ModelGraphRunner

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelPrefillRunner(ModelGraphRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        """
        定长 Prefill 前向传播 (batch, seq_len)
        通过 adapter 钩子驱动任意架构。
        """
        B, S = input_ids.shape
        embed = self.adapter.embed(self.model)
        blocks = self.adapter.blocks(self.model)
        h = embed(input_ids)

        # 初始化 cache 状态：cache_seqlens 一次性置 0，逐层 flash 写入 paged cache
        cache_lens = cache_manager._cache_seqlens_buffer[:batch_size].zero_()
        block_table = cache_manager._block_table_buffer[:batch_size]

        for layer_idx in range(self.num_layers):
            block = blocks[layer_idx]
            h = self.adapter.prefill_layer(block, h, layer_idx, B, S, self, cache_manager, block_table)

        h = self.adapter.final_norm(self.model)(h)
        return self.adapter.lm_head(self.model)(h)
