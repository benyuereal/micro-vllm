from typing import Dict, List, Tuple, Optional
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer  # 推荐导入方式


class KVCache:
    def __init__(self):
        self.seq_kv_cache = {}  # seq_id -> DynamicCache

    def allocate(self, seq_id: int, past_key_values: "DynamicCache"):
        self.seq_kv_cache[seq_id] = past_key_values

    def delete(self, seq_id: int):
        """删除指定序列的缓存"""
        print(f"seq {seq_id}: delete")
        if seq_id in self.seq_kv_cache:
            cache = self.seq_kv_cache[seq_id]
            if hasattr(cache, "layers"):
                for layer in cache.layers:
                    if isinstance(layer.keys, torch.Tensor):
                        layer.keys = layer.keys.cpu()
                    if isinstance(layer.values, torch.Tensor):
                        layer.values = layer.values.cpu()
            del self.seq_kv_cache[seq_id]

    def get(self, seq_id: int) -> Optional["DynamicCache"]:
        return self.seq_kv_cache.get(seq_id)

    def batch_kv(self, seq_ids: List[int]) -> "DynamicCache":
        batch_cache = DynamicCache()
        for seq_id in seq_ids:
            seq_cache = self.get(seq_id)
            if seq_cache is None:
                raise RuntimeError(f"seq_id {seq_id} not found in cache")

            # 兼容元组和 DynamicCache
            if isinstance(seq_cache, tuple):
                # 假设格式为 (key_0, value_0, key_1, value_1, ...)
                num_layers = len(seq_cache) // 2
                seq_layers = []
                for i in range(num_layers):
                    key = seq_cache[i * 2]
                    value = seq_cache[i * 2 + 1]
                    # 解包元组（如果 key/value 是元组）
                    if isinstance(key, tuple):
                        key = key[0]
                    if isinstance(value, tuple):
                        value = value[0]
                    layer = DynamicLayer()
                    layer.keys = key
                    layer.values = value
                    seq_layers.append(layer)
            else:
                seq_layers = seq_cache.layers

            for layer_idx in range(len(seq_layers)):
                layer = seq_layers[layer_idx]
                key = layer.keys
                value = layer.values

                # 确保 key/value 是张量（解包元组）
                if isinstance(key, tuple):
                    key = key[0]
                if isinstance(value, tuple):
                    value = value[0]

                if layer_idx >= len(batch_cache.layers):
                    new_layer = DynamicLayer()
                    new_layer.keys = key
                    new_layer.values = value
                    batch_cache.layers.append(new_layer)
                else:
                    batch_layer = batch_cache.layers[layer_idx]
                    # 确保 batch_layer.keys/values 是张量
                    if isinstance(batch_layer.keys, tuple):
                        batch_layer.keys = batch_layer.keys[0]
                    if isinstance(batch_layer.values, tuple):
                        batch_layer.values = batch_layer.values[0]
                    # 现在可以安全调用 torch.cat
                    batch_layer.keys = torch.cat([batch_layer.keys, key], dim=0)
                    batch_layer.values = torch.cat([batch_layer.values, value], dim=0)

        return batch_cache

    def unbatch_kv(self, seq_ids: List[int], batch_cache: "DynamicCache") -> Dict[int, "DynamicCache"]:
        kv_dict = {}
        batch_size = len(seq_ids)

        # 兼容元组和 DynamicCache
        if isinstance(batch_cache, tuple):
            num_layers = len(batch_cache) // 2
            layers = []
            for i in range(num_layers):
                key = batch_cache[i * 2]
                value = batch_cache[i * 2 + 1]
                # 解包元组
                if isinstance(key, tuple):
                    key = key[0]
                if isinstance(value, tuple):
                    value = value[0]
                layer = DynamicLayer()
                layer.keys = key
                layer.values = value
                layers.append(layer)
        else:
            layers = batch_cache.layers

        for i, seq_id in enumerate(seq_ids):
            seq_cache = DynamicCache()
            for layer_idx in range(len(layers)):
                layer = layers[layer_idx]
                key = layer.keys
                value = layer.values
                # 解包元组
                if isinstance(key, tuple):
                    key = key[0]
                if isinstance(value, tuple):
                    value = value[0]
                # 切片
                key = key[i:i + 1]
                value = value[i:i + 1]

                new_layer = DynamicLayer()
                new_layer.keys = key
                new_layer.values = value
                seq_cache.layers.append(new_layer)

            # 复制 cumulative_length
            if hasattr(batch_cache, "cumulative_length"):
                seq_cache.cumulative_length = batch_cache.cumulative_length.copy()

            kv_dict[seq_id] = seq_cache

        return kv_dict
