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

    @staticmethod
    def prepare_batch_inputs(
            model: PreTrainedModel,
            batch: List[Tuple[torch.Tensor, Optional[Tuple]]]
    ) -> Dict:
        """
        为Qwen模型准备批量输入格式，支持不同长度的序列
        """
        # 检查批次类型（预填充或解码）
        is_prefill = batch[0][1] is None

        input_ids_list = []
        position_ids_list = []
        attention_mask_list = []
        past_key_values_list = []
        seq_lengths = []

        # 第一步：处理每个序列的输入
        for input_tensor, past in batch:
            batch_size, seq_length = input_tensor.shape

            if is_prefill:
                # 预填充阶段
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_tensor.device)
                position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

                # 创建因果注意力掩码
                attention_mask = torch.ones((batch_size, seq_length, seq_length), device=input_tensor.device)
                attention_mask = torch.tril(attention_mask)  # 下三角矩阵（因果掩码）
            else:
                # 解码阶段
                position_ids = torch.tensor([[seq_length - 1]], dtype=torch.long, device=input_tensor.device)
                attention_mask = None  # 解码阶段不需要全掩码

            input_ids_list.append(input_tensor)
            position_ids_list.append(position_ids)
            attention_mask_list.append(attention_mask)
            past_key_values_list.append(past)
            seq_lengths.append(seq_length)

        # 第二步：对齐不同长度的序列（仅预填充阶段需要）
        if is_prefill:
            max_len = max(seq_lengths)
            padded_input_ids = []
            padded_position_ids = []
            padded_attention_mask = []

            pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0

            for i, (input_ids, position_ids, attention_mask) in enumerate(zip(
                    input_ids_list, position_ids_list, attention_mask_list
            )):
                current_len = input_ids.shape[1]
                pad_len = max_len - current_len

                # 填充input_ids
                if pad_len > 0:
                    padded_input = torch.nn.functional.pad(
                        input_ids,
                        (0, pad_len),
                        value=pad_token_id
                    )
                else:
                    padded_input = input_ids
                padded_input_ids.append(padded_input)

                # 填充position_ids
                if pad_len > 0:
                    padded_pos = torch.nn.functional.pad(
                        position_ids,
                        (0, pad_len),
                        value=0  # 填充位置用0
                    )
                else:
                    padded_pos = position_ids
                padded_position_ids.append(padded_pos)

                # 填充attention_mask (仅当需要填充时)
                if attention_mask is not None:
                    if pad_len > 0:
                        # 水平填充 (右侧)
                        row_pad = torch.zeros((batch_size, current_len, pad_len), device=input_ids.device)
                        padded_attn_h = torch.cat([attention_mask, row_pad], dim=2)

                        # 垂直填充 (底部)
                        col_pad = torch.zeros((batch_size, pad_len, max_len), device=input_ids.device)
                        padded_attn = torch.cat([padded_attn_h, col_pad], dim=1)
                    else:
                        padded_attn = attention_mask
                    padded_attention_mask.append(padded_attn)
                else:
                    padded_attention_mask.append(None)

            # 合并所有序列
            input_ids = torch.cat(padded_input_ids, dim=0)
            position_ids = torch.cat(padded_position_ids, dim=0)
            attention_mask = torch.cat(padded_attention_mask, dim=0) if any(
                m is not None for m in padded_attention_mask) else None
        else:
            # 解码阶段不需要填充
            input_ids = torch.cat(input_ids_list, dim=0)
            position_ids = torch.cat(position_ids_list, dim=0)
            attention_mask = None

        # 关键修复：确保past_key_values格式正确
        if is_prefill:
            # 预填充阶段不需要past_key_values
            past_key_values = None
        else:
            # 解码阶段需要传递正确的past_key_values结构
            # 确保每个序列的past_key_values是元组格式
            past_key_values = []
            for past in past_key_values_list:
                if past is not None:
                    past_key_values.append(past)
                else:
                    # 对于没有缓存的情况，创建一个空元组占位
                    past_key_values.append(())

            # 如果所有序列都没有缓存，则设为None
            if all(len(p) == 0 for p in past_key_values):
                past_key_values = None

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True
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
        start_idx = 0

        for i, seq_len in enumerate(seq_lengths):
            # 提取当前序列对应的输出
            end_idx = start_idx + 1
            seq_logits = logits[start_idx:end_idx, -1:, :]  # 只取最后一个token

            # 提取当前序列对应的KV缓存
            seq_past = []
            for layer in past_key_values:
                seq_past.append((
                    layer[0][start_idx:end_idx],
                    layer[1][start_idx:end_idx]
                ))

            results.append((seq_logits, tuple(seq_past)))
            start_idx = end_idx

        return results