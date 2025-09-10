import torch
from transformers import PreTrainedModel, DynamicCache
from typing import Dict, Optional, List, Tuple


class QwenModelAdapter:
    @staticmethod
    def prepare_inputs(
            model: PreTrainedModel,
            input_ids: torch.Tensor,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict:
        batch_size, seq_length = input_ids.shape
        cache = past_key_values
        if past_key_values is None:
            # Prefill
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        else:
            # Decode
            current_seq_len = past_key_values[0][0].size(1)  # 从第一个层的key获取长度
            position_ids = torch.full((batch_size, 1), current_seq_len, dtype=torch.long, device=input_ids.device)
            attention_mask = None
            # 将普通列表转换为 DynamicCache
            cache = DynamicCache()

            # 逐层处理
            for layer_idx, layer_kv in enumerate(past_key_values):
                # 每层是一个元组 (key_states, value_states)
                key_states, value_states = layer_kv

                # 调用update方法更新缓存
                # 注意：这里不需要传递cache_kwargs，除非有特殊需求
                cache.update(
                    key_states=key_states,
                    value_states=value_states,
                    layer_idx=layer_idx,
                    cache_kwargs=None  # 通常可以为None
                )


        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": True
        }

    @staticmethod
    def process_outputs(
            outputs: Tuple,
            input_length: int
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        logits = outputs.logits
        past_key_values = outputs.past_key_values  # 已经是元组形式

        # 转换为层级的KV缓存列表
        kv_cache = []
        for layer_past in past_key_values:
            k, v = layer_past
            kv_cache.append((k, v))

        if input_length > 1:
            logits = logits[:, -1:, :]

        return logits, kv_cache


