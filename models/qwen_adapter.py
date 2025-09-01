import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, Tuple, List


class QwenModelAdapter:
    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            past_key_values: Optional[Tuple] = None,
            sequence_lengths: Optional[List[int]] = None
    ) -> Dict:
        """
        为Qwen模型准备输入格式
        """
        batch_size, seq_length = input_ids.shape

        # 验证past_key_values结构
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple), "past_key_values must be a tuple"
            for layer_kv in past_key_values:
                assert isinstance(layer_kv, tuple) and len(
                    layer_kv) == 2, "Each layer should be a tuple of (key, value)"

        if past_key_values is None:
            # 首次推理（prefill）
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        else:
            # 增量推理（decode）
            if sequence_lengths is None:
                raise ValueError("sequence_lengths must be provided for decoding")

            # 创建位置ID：每个序列的当前位置
            position_ids = torch.tensor(
                [[length] for length in sequence_lengths],  # 使用完整序列长度
                dtype=torch.long,
                device=input_ids.device
            )
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

        # 验证past_key_values结构
        assert isinstance(past_key_values, tuple), "past_key_values must be a tuple"
        for i, layer_kv in enumerate(past_key_values):
            assert isinstance(layer_kv, tuple) and len(layer_kv) == 2, f"Layer {i} should be a tuple of (key, value)"

        # 对于decode步骤，我们只需要最后一个token的logits
        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, past_key_values