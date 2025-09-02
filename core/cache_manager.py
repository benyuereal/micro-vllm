# core/cache_manager.py
from typing import Dict, List, Tuple, Optional
import torch
from transformers import DynamicCache, DynamicLayer


class KVCache:
    def __init__(self):
        self.seq_kv_cache = {}  # seq_id -> DynamicCache

    def allocate(self, seq_id: int, past_key_values: "DynamicCache"):
        self.seq_kv_cache[seq_id] = past_key_values

    def get(self, seq_id: int) -> Optional["DynamicCache"]:
        return self.seq_kv_cache.get(seq_id)

    def batch_kv(self, seq_ids: List[int]) -> "DynamicCache":
        batch_cache = DynamicCache()

        for seq_id in seq_ids:
            seq_cache = self.get(seq_id)
            if seq_cache is None:
                raise RuntimeError(f"seq_id {seq_id} not found in cache")

            # ✅ 用 .layers + .keys / .values（v4.56.0）
            for layer_idx in range(len(seq_cache.layers)):
                layer = seq_cache.layers[layer_idx]
                key = layer.keys  # [1, num_heads, seq_len, head_dim]
                value = layer.values

                if layer_idx >= len(batch_cache.layers):
                    # 第一次添加该 layer
                    new_layer = DynamicLayer()
                    new_layer.keys = key
                    new_layer.values = value
                    batch_cache.layers.append(new_layer)
                else:
                    # 后续合并：在 batch 维度拼接
                    batch_layer = batch_cache.layers[layer_idx]
                    batch_layer.keys = torch.cat([batch_layer.keys, key], dim=0)
                    batch_layer.values = torch.cat([batch_layer.values, value], dim=0)

        # ✅ 更新 cumulative_length
        if hasattr(seq_cache, "cumulative_length"):
            batch_cache.cumulative_length = seq_cache.cumulative_length
        return batch_cache

    def unbatch_kv(self, seq_ids: List[int], batch_cache: "DynamicCache") -> Dict[int, "DynamicCache"]:
        kv_dict = {}
        batch_size = len(seq_ids)

        for i, seq_id in enumerate(seq_ids):
            seq_cache = DynamicCache()

            for layer_idx in range(len(batch_cache.layers)):
                layer = batch_cache.layers[layer_idx]
                key = layer.keys[i:i + 1]  # [1, num_heads, seq_len, head_dim]
                value = layer.values[i:i + 1]

                # 创建新的 DynamicLayer
                new_layer = DynamicLayer()
                new_layer.keys = key
                new_layer.values = value
                seq_cache.layers.append(new_layer)

            # ✅ 更新 cumulative_length
            if hasattr(batch_cache, "cumulative_length"):
                seq_cache.cumulative_length = batch_cache.cumulative_length
            kv_dict[seq_id] = seq_cache

        return kv_dict








