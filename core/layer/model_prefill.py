import torch
import logging
from .model_graph import ModelGraphRunner
from models.base import PrefillMeta

logger = logging.getLogger(__name__)


class ModelPrefillRunner(ModelGraphRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids: torch.Tensor, cache_manager, meta: PrefillMeta) -> torch.Tensor:
        """变长 Prefill 前向：input_ids 为 1D [total_tokens]，batch 内各 seq 长度可不同。
        用 cu_seqlens 掩码（flash_attn_varlen_func）+ block_table 读 paged cache。
        通过 adapter.prefill 钩子驱动任意架构。"""
        embed = self.adapter.embed(self.model)
        blocks = self.adapter.blocks(self.model)
        h = embed(input_ids)  # [total_tokens, hidden]

        for layer_idx in range(self.num_layers):
            block = blocks[layer_idx]
            h = self.adapter.prefill(block, h, layer_idx, self, cache_manager, meta)

        h = self.adapter.final_norm(self.model)(h)
        return self.adapter.lm_head(self.model)(h)
