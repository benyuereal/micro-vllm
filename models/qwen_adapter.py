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
            position_ids = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        else:
            # 动态生成位置ID和注意力掩码
            past_length = past_key_values.get_seq_length()
            position_ids = torch.arange(past_length, past_length + seq_length, device=input_ids.device).unsqueeze(0)

            # 创建因果注意力掩码
            attention_mask = torch.ones((batch_size, past_length + seq_length), device=input_ids.device)
            for i in range(batch_size):
                attention_mask[i, :past_length] = 1  # 历史token
                attention_mask[i, past_length:] = torch.tril(torch.ones(seq_length, seq_length))[0]

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