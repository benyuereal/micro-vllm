# core/cache_manager.py
from typing import Dict, List, Tuple, Optional
import torch
from transformers import DynamicCache, DynamicLayer


class KVCache:
    def __init__(self):
        self.seq_kv_cache = {}  # seq_id -> DynamicCache

    def allocate(self, seq_id: int, past_key_values: "DynamicCache"):
        self.seq_kv_cache[seq_id] = past_key_values

    def delete(self, seq_id: int):
        """删除指定序列的缓存"""
        print(f"seq {seq_id}: delete")
        if seq_id in self.seq_kv_cache:
            # 释放显存（如果有GPU内存占用）
            cache = self.seq_kv_cache[seq_id]
            for layer in cache.layers:
                if isinstance(layer.keys, torch.Tensor):
                    layer.keys = layer.keys.cpu()
                if isinstance(layer.values, torch.Tensor):
                    layer.values = layer.values.cpu()

            # 从缓存中移除
            del self.seq_kv_cache[seq_id]

    def get(self, seq_id: int) -> Optional["DynamicCache"]:
        return self.seq_kv_cache.get(seq_id)

    def batch_kv(self, seq_ids: List[int]) -> "DynamicCache":
        batch_cache = DynamicCache()

        for seq_id in seq_ids:
            seq_cache = self.get(seq_id)
            if seq_cache is None:
                raise RuntimeError(f"seq_id {seq_id} not found in cache")

            for layer_idx in range(len(seq_cache.layers)):
                layer = seq_cache.layers[layer_idx]
                key = layer.keys  # [1, num_heads, seq_len, head_dim]
                value = layer.values

                if layer_idx >= len(batch_cache.layers):
                    # 第一次添加该层
                    new_layer = DynamicLayer()
                    new_layer.keys = key
                    new_layer.values = value
                    batch_cache.layers.append(new_layer)
                else:
                    # 获取已存在的层
                    batch_layer = batch_cache.layers[layer_idx]

                    # 修复点：检查是否为首次添加值
                    if batch_layer.keys is None:
                        # 首次添加直接赋值
                        batch_layer.keys = key
                        batch_layer.values = value
                    else:
                        # 后续添加进行拼接
                        batch_layer.keys = torch.cat([batch_layer.keys, key], dim=0)
                        batch_layer.values = torch.cat([batch_layer.values, value], dim=0)

        return batch_cache

    def unbatch_kv(self, seq_ids: List[int], batch_cache: "DynamicCache") -> Dict[int, "DynamicCache"]:
        kv_dict = {}
        batch_size = len(seq_ids)

        # 兼容元组和 DynamicCache
        if isinstance(batch_cache, tuple):
            # 假设格式为 (key_0, value_0, key_1, value_1, ...)
            num_layers = len(batch_cache) // 2
            layers = []
            for i in range(num_layers):
                key = batch_cache[i * 2]
                value = batch_cache[i * 2 + 1]
                layer = DynamicLayer()
                layer.keys = key
                layer.values = value
                layers.append(layer)
        else:
            layers = batch_cache.layers  # 直接使用 DynamicCache

        for i, seq_id in enumerate(seq_ids):
            seq_cache = DynamicCache()
            for layer_idx in range(len(layers)):
                layer = layers[layer_idx]
                key = layer.keys[i:i + 1]
                value = layer.values[i:i + 1]

                # 创建新的 DynamicLayer
                new_layer = DynamicLayer()
                new_layer.keys = key
                new_layer.values = value
                seq_cache.layers.append(new_layer)

            # ✅ 修复：正确设置cumulative_length
            if hasattr(batch_cache, "cumulative_length"):
                # 复制原始累计长度
                seq_cache.cumulative_length = batch_cache.cumulative_length.copy()

            kv_dict[seq_id] = seq_cache

        return kv_dict









