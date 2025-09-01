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
        为批量请求准备输入
        """
        input_ids = []
        position_ids = []
        attention_masks = []
        past_key_values = []
        seq_lengths = []

        for input_tensor, past in batch:
            if past is None:
                # 预填充请求
                seq_len = input_tensor.shape[1]
                pos_ids = torch.arange(0, seq_len, device=input_tensor.device).unsqueeze(0)
                attn_mask = torch.ones((1, seq_len, seq_len), device=input_tensor.device)
                attn_mask = torch.tril(attn_mask)  # 因果注意力掩码
            else:
                # 解码请求
                seq_len = past[0][0].shape[2] + 1  # 历史长度 + 当前token
                pos_ids = torch.tensor([[seq_len - 1]], device=input_tensor.device)
                attn_mask = None

            input_ids.append(input_tensor)
            position_ids.append(pos_ids)
            attention_masks.append(attn_mask)
            past_key_values.append(past)
            seq_lengths.append(seq_len)

        # 填充输入使其长度一致
        max_len = max(seq_lengths)
        padded_inputs = []
        padded_attentions = []

        for i, inp in enumerate(input_ids):
            pad_len = max_len - inp.shape[1]
            padded = torch.nn.functional.pad(inp, (0, pad_len), value=model.config.pad_token_id)
            padded_inputs.append(padded)

            if attention_masks[i] is not None:
                attn_pad = torch.zeros((1, pad_len, max_len), device=inp.device)
                padded_attn = torch.cat([
                    attention_masks[i],
                    torch.zeros((1, seq_lengths[i], pad_len), device=inp.device)
                ], dim=2)
                padded_attn = torch.cat([padded_attn, attn_pad], dim=1)
                padded_attentions.append(padded_attn)
            else:
                padded_attentions.append(None)

        return {
            "input_ids": torch.cat(padded_inputs, dim=0),
            "position_ids": torch.cat(position_ids, dim=0),
            "attention_mask": torch.cat(padded_attentions, dim=0) if any(
                m is not None for m in padded_attentions) else None,
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