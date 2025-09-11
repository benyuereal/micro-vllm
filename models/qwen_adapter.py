import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, List, Tuple

from core import KVCache


class QwenModelAdapter:
    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            cache_manager: Optional[KVCache] = None,
            seq_ids: Optional[List[int]] = None
    ) -> Dict:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if cache_manager is None or seq_ids is None:
            # Prefill阶段
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), device=device)
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "use_cache": True
            }
        else:
            # Decode阶段（使用PageAttention）
            # 获取序列的token位置
            token_positions = []
            for seq_id in seq_ids:
                tokens = cache_manager.get_sequence_tokens(seq_id)
                positions = [pos for _, pos in tokens]
                token_positions.append(positions)

            return {
                "input_ids": input_ids,
                "cache_manager": cache_manager,
                "seq_ids": seq_ids,
                "token_positions": token_positions,
                "use_cache": True
            }

    @staticmethod
    def process_outputs(
            outputs: Tuple,
            input_length: int
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, past_key_values

    @staticmethod
    def extract_kv_for_token(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            token_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从KV缓存中提取特定token的KV数据"""
        layer_k = []
        layer_v = []

        for layer_idx, (k, v) in enumerate(past_key_values):
            layer_k.append(k[:, :, token_idx:token_idx + 1, :])
            layer_v.append(v[:, :, token_idx:token_idx + 1, :])

        return torch.cat(layer_k, dim=0), torch.cat(layer_v, dim=0)