import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, Tuple


class QwenModelAdapter:
    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            past_key_values: Optional[Tuple] = None
    ) -> Dict:
        """
        为Qwen模型准备输入格式
        """
        batch_size, seq_length = input_ids.shape

        # 处理位置编码
        if past_key_values is None:
            # 首次推理（prefill）
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        else:
            # 增量推理（decode）
            # 对于解码，我们只需要当前位置
            position_ids = torch.tensor([[seq_length - 1]], dtype=torch.long, device=input_ids.device)
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

        # 对于decode步骤，我们只需要最后一个token的logits
        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, past_key_values