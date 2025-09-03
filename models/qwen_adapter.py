import torch
from transformers import PreTrainedModel, DynamicCache
from typing import Dict, Optional, Tuple


class QwenModelAdapter:

    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            past_key_values: Optional["DynamicCache"] = None
    ) -> Dict:
        batch_size, seq_length = input_ids.shape

        if past_key_values is None:
            # Prefill
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        else:
            # Decode: 用 DynamicCache.get_seq_length()
            current_seq_len = past_key_values.get_seq_length()  # ✅ 正确获取长度
            position_ids = torch.full((batch_size, 1), current_seq_len, dtype=torch.long, device=input_ids.device)
            attention_mask = None

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True
        }

    @staticmethod
    def process_outputs(
            outputs: Tuple,
            input_length: int
    ) -> Tuple[torch.Tensor, DynamicCache]:  # 修改返回类型
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # 将元组格式转换为DynamicCache
        cache = DynamicCache.from_legacy_cache(past_key_values)

        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, cache  # 返回DynamicCache