import torch
from transformers import PreTrainedModel
from typing import Dict, Optional, Tuple, List

from core import KVCache


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
            input_ids: torch.Tensor,
            seq_metas: List[Dict],
            cache_manager: KVCache
    ) -> Dict:
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 获取序列ID列表
        seq_ids = [meta["seq_id"] for meta in seq_metas]

        # 从缓存管理器批量获取缓存
        cache_entries = cache_manager.get_batch(seq_ids)
        past_key_values = [entry["past_key_values"] if entry else None for entry in cache_entries]

        # 合并缓存
        merged_cache = QwenModelAdapter.merge_caches(past_key_values)

        # 构建位置ID和注意力掩码
        position_ids = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        # 当前序列长度（不包括新token）
        current_lengths = []
        for i, meta in enumerate(seq_metas):
            # 计算当前序列的上下文长度
            if meta["is_prefill"]:
                # 首次推理：位置从0开始
                seq_len = input_ids.shape[1]
                position_ids[i] = torch.arange(0, seq_len, device=device)
                current_length = 0
            else:
                # 增量推理：位置从已有长度开始
                cache_entry = cache_entries[i]
                if cache_entry and cache_entry["past_key_values"] is not None:
                    # 取第一层key的长度
                    current_length = cache_entry["past_key_values"][0][0].shape[2]
                else:
                    current_length = 0
                position_ids[i, -1] = current_length  # 仅当前token的位置
                current_lengths.append(current_length)

        # 创建注意力掩码（考虑缓存）
        if merged_cache is not None:
            # 获取合并后缓存的序列长度
            ref_cache = merged_cache[0][0]  # 第一层的key
            total_length = ref_cache.shape[2] + 1  # 缓存长度 + 当前token

            # 创建因果注意力掩码
            attention_mask = torch.tril(
                torch.ones((batch_size, total_length, total_length), device=device)
            )

            # 仅保留有效的上下文部分
            valid_lengths = [length + 1 for length in current_lengths]
            for i, valid_len in enumerate(valid_lengths):
                attention_mask[i, :, valid_len:] = 0  # 屏蔽未来token
        else:
            # 没有缓存时使用完整因果掩码
            seq_len = input_ids.shape[1]
            attention_mask = torch.tril(
                torch.ones((batch_size, seq_len, seq_len), device=device)
            )

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": merged_cache,
            "use_cache": True
        }

    # 在 QwenModelAdapter 类中添加
    @staticmethod
    def merge_caches(caches: List[Optional[Tuple]]) -> Tuple:
        """
        合并多个序列的KV缓存为一个批处理缓存
        处理不同长度的序列缓存，通过填充对齐
        """
        if not caches or all(cache is None for cache in caches):
            return None

        # 确定最大序列长度
        max_lengths = []
        for cache in caches:
            if cache is not None:
                # 取第一层作为参考（所有层长度相同）
                layer_cache = cache[0]
                if layer_cache[0] is not None:  # key
                    max_lengths.append(layer_cache[0].shape[2])
                else:
                    max_lengths.append(0)
        max_seq_len = max(max_lengths) if max_lengths else 0

        # 如果没有有效缓存，返回None
        if max_seq_len == 0:
            return None

        # 确定层数和头维度
        num_layers = len(caches[0]) if caches[0] is not None else 0
        if num_layers == 0:
            return None

        # 获取第一个有效缓存的维度作为参考
        ref_cache = next((c for c in caches if c is not None), None)
        _, num_heads, _, head_dim = ref_cache[0][0].shape if ref_cache[0][0] is not None else (0, 0, 0, 0)

        # 初始化合并后的缓存结构
        merged_cache = []
        for layer_idx in range(num_layers):
            merged_keys, merged_values = [], []

            for i, cache in enumerate(caches):
                if cache is None or cache[layer_idx] is None:
                    # 创建空缓存
                    empty_key = torch.zeros(
                        (1, num_heads, max_seq_len, head_dim),
                        dtype=ref_cache[0][0].dtype,
                        device=ref_cache[0][0].device
                    )
                    empty_value = torch.zeros(
                        (1, num_heads, max_seq_len, head_dim),
                        dtype=ref_cache[0][1].dtype,
                        device=ref_cache[0][1].device
                    )
                    merged_keys.append(empty_key)
                    merged_values.append(empty_value)
                else:
                    key, value = cache[layer_idx]
                    seq_len = key.shape[2]

                    # 填充到最大长度
                    if seq_len < max_seq_len:
                        pad_len = max_seq_len - seq_len
                        padded_key = F.pad(key, (0, 0, 0, pad_len), "constant", 0)
                        padded_value = F.pad(value, (0, 0, 0, pad_len), "constant", 0)
                        merged_keys.append(padded_key)
                        merged_values.append(padded_value)
                    else:
                        merged_keys.append(key)
                        merged_values.append(value)

            # 沿批处理维度拼接
            merged_keys_batch = torch.cat(merged_keys, dim=0)
            merged_values_batch = torch.cat(merged_values, dim=0)
            merged_cache.append((merged_keys_batch, merged_values_batch))

        return tuple(merged_cache)