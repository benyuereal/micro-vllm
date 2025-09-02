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

    # core/cache_manager.py
    def batch_kv(self, seq_ids: List[int]) -> Tuple[torch.Tensor, ...]:
        """
        将多个序列的 past_key_values 按 layer 拼接成 batch
        输入: seq_ids = [id1, id2, ...]
        输出: tuple of (batch_k, batch_v) for each layer
        """
        # 获取每个序列的 past_key_values
        all_seq_kv = [self.get(seq_id) for seq_id in seq_ids]  # [((k0,v0), (k1,v1), ...), ...]

        # 检查是否为空
        if not all_seq_kv or all_seq_kv[0] is None:
            return None

        # 按 layer 分组：将第 i 层的 (k, v) 收集起来
        num_layers = len(all_seq_kv[0])
        batch_kv = []

        # 在 batch_kv 中
        print(f"Batching {len(seq_ids)} sequences, num_layers={len(all_seq_kv[0])}")
        for i, seq_kv in enumerate(all_seq_kv):
            print(f"  seq {seq_ids[i]}: {len(seq_kv)} layers, first key shape: {seq_kv[0][0].shape}")


        for layer_idx in range(num_layers):
            keys = []
            values = []
            for seq_kv in all_seq_kv:
                k, v = seq_kv[layer_idx]  # ✅ 正确解包每个序列的第 layer_idx 层
                keys.append(k)
                values.append(v)
            # 拼接 batch
            batch_k = torch.cat(keys, dim=0)  # [batch_size, ...]
            batch_v = torch.cat(values, dim=0)
            batch_kv.append((batch_k, batch_v))

        return tuple(batch_kv)  # ((k0_batch, v0_batch), (k1_batch, v1_batch), ...)

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
                # 取第 i 个序列的 k 和 v
                k_i = k[i:i + 1]  # 保持 batch dim
                v_i = v[i:i + 1]
                kv_per_seq.append((k_i, v_i))
            kv_dict[seq_id] = tuple(kv_per_seq)

        return kv_dict

