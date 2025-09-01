# qwen_adapter.py
import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, Tuple, List, Union


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
            # 创建因果注意力掩码
            attention_mask = torch.ones((batch_size, seq_length, seq_length), device=input_ids.device)
            attention_mask = torch.tril(attention_mask)  # 下三角掩码
        else:
            # 增量推理（decode）
            # 对于解码，我们只需要当前位置
            position_ids = torch.tensor([[seq_length - 1]], dtype=torch.long, device=input_ids.device)
            # 对于解码阶段，使用None让模型自动处理注意力掩码
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

    @staticmethod
    def prepare_batch_inputs(
            model: PreTrainedModel,
            batch: List[Tuple[torch.Tensor, Optional[Tuple]]]
    ) -> Dict:
        """
        为批量请求准备输入
        """
        input_ids_list = []
        position_ids_list = []
        attention_mask_list = []
        past_key_values_list = []

        # 分离预填充和解码请求
        prefill_requests = []
        decode_requests = []

        for i, (input_tensor, past) in enumerate(batch):
            if past is None:
                prefill_requests.append((i, input_tensor))
            else:
                decode_requests.append((i, input_tensor, past))

        # 分别处理预填充和解码请求
        if prefill_requests:
            prefill_inputs = QwenModelAdapter._prepare_prefill_batch(model, prefill_requests)
            input_ids_list.append(prefill_inputs["input_ids"])
            position_ids_list.append(prefill_inputs["position_ids"])
            attention_mask_list.append(prefill_inputs["attention_mask"])
            past_key_values_list.extend(prefill_inputs["past_key_values"])

        if decode_requests:
            decode_inputs = QwenModelAdapter._prepare_decode_batch(model, decode_requests)
            input_ids_list.append(decode_inputs["input_ids"])
            position_ids_list.append(decode_inputs["position_ids"])
            # 解码阶段注意力掩码为None
            if decode_inputs["attention_mask"] is not None:
                attention_mask_list.append(decode_inputs["attention_mask"])
            past_key_values_list.extend(decode_inputs["past_key_values"])

        # 合并所有输入
        input_ids = torch.cat(input_ids_list, dim=0) if input_ids_list else torch.empty(0)

        # 处理position_ids - 需要单独处理每个请求
        position_ids = position_ids_list[0] if len(position_ids_list) == 1 else None
        if len(position_ids_list) > 1:
            # 对于混合批次，需要在模型内部处理position_ids
            position_ids = None

        # 处理attention_mask
        attention_mask = torch.cat(attention_mask_list, dim=0) if attention_mask_list else None

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values_list if past_key_values_list else None,
            "use_cache": True
        }

    @staticmethod
    def _prepare_prefill_batch(model: PreTrainedModel, requests: List[Tuple[int, torch.Tensor]]) -> Dict:
        """处理预填充批次"""
        input_ids = []
        position_ids = []
        attention_masks = []
        past_key_values = [None] * len(requests)

        max_seq_len = max(input_tensor.shape[1] for _, input_tensor in requests)

        for idx, input_tensor in requests:
            seq_len = input_tensor.shape[1]

            # 填充input_ids
            padded_input = torch.nn.functional.pad(
                input_tensor,
                (0, max_seq_len - seq_len),
                value=model.config.pad_token_id
            )
            input_ids.append(padded_input)

            # 创建position_ids
            pos_ids = torch.arange(0, max_seq_len, device=input_tensor.device)
            pos_ids = pos_ids.unsqueeze(0)
            position_ids.append(pos_ids)

            # 创建注意力掩码
            attn_mask = torch.ones((1, max_seq_len, max_seq_len), device=input_tensor.device)
            attn_mask = torch.tril(attn_mask)  # 因果掩码
            # 屏蔽填充部分
            attn_mask[:, :, seq_len:] = 0
            attn_mask[:, seq_len:, :] = 0
            attention_masks.append(attn_mask)

        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "position_ids": torch.cat(position_ids, dim=0),
            "attention_mask": torch.cat(attention_masks, dim=0),
            "past_key_values": past_key_values
        }

    @staticmethod
    def _prepare_decode_batch(model: PreTrainedModel, requests: List[Tuple[int, torch.Tensor, Tuple]]) -> Dict:
        """处理解码批次"""
        input_ids = torch.cat([input_tensor for _, input_tensor, _ in requests], dim=0)

        # 对于解码，position_ids都是当前位置
        position_ids = torch.tensor([
            [past[0][0].shape[2]] for _, _, past in requests
        ], device=input_ids.device, dtype=torch.long)

        # 解码阶段不需要显式的注意力掩码
        attention_mask = None
        past_key_values = [past for _, _, past in requests]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values
        }

    @staticmethod
    def process_batch_outputs(
            outputs: Tuple,
            seq_lengths: List[int]
    ) -> List[Tuple[torch.Tensor, Tuple]]:
        """
        处理批量输出结果
        """
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        results = []
        batch_size = logits.shape[0]

        for i in range(batch_size):
            # 提取当前序列对应的输出
            seq_logits = logits[i:i + 1, -1:, :]  # 只取最后一个token

            # 提取当前序列对应的KV缓存
            seq_past = []
            for layer_idx in range(len(past_key_values)):
                k_cache = past_key_values[layer_idx][0][i:i + 1] if past_key_values[layer_idx][0] is not None else None
                v_cache = past_key_values[layer_idx][1][i:i + 1] if past_key_values[layer_idx][1] is not None else None
                seq_past.append((k_cache, v_cache))

            results.append((seq_logits, tuple(seq_past)))

        return results