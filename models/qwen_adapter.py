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
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        处理Qwen模型的输出
        """
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            # 模型没有返回 past_key_values，可能是 use_cache=False 或 bug
            raise RuntimeError("Model did not return past_key_values. Check use_cache=True and model config.")

        # 对于decode步骤，我们只需要最后一个token的logits
        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, past_key_values