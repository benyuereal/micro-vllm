# core/cache_manager.py
from typing import Dict, List, Tuple
import torch

class KVCache:
    def __init__(self):
        self.seq_kv_cache: Dict[int, Tuple] = {}  # seq_id -> past_key_values
        self.max_batch_size = 8

    def allocate(self, seq_id: int, kv: Tuple):
        self.seq_kv_cache[seq_id] = kv

    def get(self, seq_id: int):
        return self.seq_kv_cache.get(seq_id)

    def delete(self, seq_id: int):
        if seq_id in self.seq_kv_cache:
            del self.seq_kv_cache[seq_id]

    def batch_kv(self, seq_ids: List[int]) -> Tuple[torch.Tensor, ...]:
        """
        将多个序列的 past_key_values 拼接成 batch
        返回: (batch_past_key_values) 每个 layer 是一个 tuple (key, value)
        """
        batch_kv = []
        for kv_list in zip(*[self.get(seq_id) for seq_id in seq_ids]):
            # kv_list: [(k1, v1), (k2, v2), ...]
            keys = torch.cat([k for k, v in kv_list], dim=0)  # batch_size, ...
            values = torch.cat([v for k, v in kv_list], dim=0)
            batch_kv.append((keys, values))
        return tuple(batch_kv)

    def unbatch_kv(self, seq_ids: List[int], batch_kv: Tuple) -> Dict[int, Tuple]:
        """
        将 batch 的 past_key_values 拆分回每个序列
        """
        kv_dict = {}
        batch_size = len(seq_ids)
        for i, seq_id in enumerate(seq_ids):
            kv_per_seq = []
            for layer_kv in batch_kv:
                k, v = layer_kv
                kv_per_seq.append((k[i:i+1], v[i:i+1]))
            kv_dict[seq_id] = tuple(kv_per_seq)
        return kv_dict
