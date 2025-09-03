from typing import Dict, List, Tuple, Optional
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer  # 推荐导入方式


class KVCache:
    def __init__(self):
        self.seq_kv_cache = {}  # seq_id -> (DynamicCache, seq_length)

    def allocate(self, seq_id: int, past_key_values: "DynamicCache", seq_length: int):
        self.seq_kv_cache[seq_id] = (past_key_values, seq_length)

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
        seq_caches, seq_lengths = [], []

        for seq_id in seq_ids:
            cache, length = self.get(seq_id)
            if cache is None:
                raise RuntimeError(f"seq_id {seq_id} not found in cache")
            seq_caches.append(cache)
            seq_lengths.append(length)

        # 按序列长度排序（长序列在前）
        sorted_indices = sorted(range(len(seq_lengths)), key=lambda i: -seq_lengths[i])
        seq_caches = [seq_caches[i] for i in sorted_indices]
        seq_lengths = [seq_lengths[i] for i in sorted_indices]

        # 逐层拼接
        max_layers = max(len(cache.layers) for cache in seq_caches)
        for layer_idx in range(max_layers):
            all_keys, all_values = [], []

            for cache in seq_caches:
                if layer_idx < len(cache.layers):
                    layer = cache.layers[layer_idx]
                    all_keys.append(layer.keys)
                    all_values.append(layer.values)
                else:
                    # 补零对齐
                    dummy_shape = (1, layer.keys.shape[1], 0, layer.keys.shape[3])
                    all_keys.append(torch.zeros(dummy_shape, device=layer.keys.device))
                    all_values.append(torch.zeros(dummy_shape, device=layer.values.device))

            batch_layer = DynamicLayer()
            batch_layer.keys = torch.cat(all_keys, dim=0)
            batch_layer.values = torch.cat(all_values, dim=0)
            batch_cache.layers.append(batch_layer)

        return batch_cache

    def unbatch_kv(self, seq_ids: List[int], batch_cache: "DynamicCache") -> Dict[int, "DynamicCache"]:
        kv_dict = {}
        batch_size = len(seq_ids)

        for i, seq_id in enumerate(seq_ids):
            seq_cache = DynamicCache()
            for layer in batch_cache.layers:
                new_layer = DynamicLayer()
                # 精确提取对应序列的缓存
                new_layer.keys = layer.keys[i:i + 1]
                new_layer.values = layer.values[i:i + 1]
                seq_cache.layers.append(new_layer)

            kv_dict[seq_id] = seq_cache

        return kv_dict
