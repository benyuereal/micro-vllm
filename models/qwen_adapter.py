# models/qwen_adapter.py
import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, Tuple, List


class QwenModelAdapter:
    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            past_key_values: Optional[Tuple] = None
    ) -> Dict:
        """
        为Qwen模型准备输入格式（单序列）
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
    def prepare_batch_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values_list: Optional[List[Tuple]] = None
    ) -> Dict:
        """
        为Qwen模型准备批量输入格式
        """
        batch_size, seq_length = input_ids.shape

        if past_key_values_list is None:
            # Prefill阶段
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "use_cache": True
            }
        else:
            # Decode阶段 - 处理多个序列的past_key_values
            # 将past_key_values_list转换为模型需要的格式
            if past_key_values_list and past_key_values_list[0] is not None:
                # 获取past_key_values的结构信息
                num_layers = len(past_key_values_list[0])
                num_heads = past_key_values_list[0][0][0].size(1)
                head_dim = past_key_values_list[0][0][0].size(3)

                # 初始化批量的past_key_values
                batch_past_key_values = []
                for layer_idx in range(num_layers):
                    key_tensors = []
                    value_tensors = []

                    for seq_idx, past_key_values in enumerate(past_key_values_list):
                        if past_key_values is not None:
                            key_tensors.append(past_key_values[layer_idx][0])
                            value_tensors.append(past_key_values[layer_idx][1])

                    # 在批次维度上拼接
                    if key_tensors:
                        batch_key = torch.cat(key_tensors, dim=0)
                        batch_value = torch.cat(value_tensors, dim=0)
                        batch_past_key_values.append((batch_key, batch_value))
                    else:
                        batch_past_key_values.append((None, None))
            else:
                batch_past_key_values = None

            # Decode阶段每个序列只输入一个token
            position_ids = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device) * (seq_length - 1)

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": batch_past_key_values,
                "use_cache": True
            }

    @staticmethod
    def process_outputs(
            outputs: Tuple,
            input_length: int
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        处理Qwen模型的输出（单序列）
        """
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # 对于decode步骤，我们只需要最后一个token的logits
        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, past_key_values

    @staticmethod
    def process_batch_outputs(
            outputs: Tuple,
            input_lengths: List[int]
    ) -> Tuple[List[torch.Tensor], List[Tuple]]:
        """
        处理Qwen模型的批量输出
        """
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        batch_size = logits.size(0)

        # 分割logits
        result_logits = []
        start_idx = 0
        for length in input_lengths:
            end_idx = start_idx + 1  # 每个序列只取最后一个token的logits
            result_logits.append(logits[start_idx:end_idx])
            start_idx = end_idx

        # 分割past_key_values
        result_kv = []
        if past_key_values:
            num_layers = len(past_key_values)
            for i in range(batch_size):
                seq_kv = []
                for layer_idx in range(num_layers):
                    layer_k = past_key_values[layer_idx][0][i:i + 1] if past_key_values[layer_idx][
                                                                            0] is not None else None
                    layer_v = past_key_values[layer_idx][1][i:i + 1] if past_key_values[layer_idx][
                                                                            1] is not None else None
                    seq_kv.append((layer_k, layer_v))
                result_kv.append(tuple(seq_kv))
        else:
            result_kv = [None] * batch_size

        return result_logits, result_kv