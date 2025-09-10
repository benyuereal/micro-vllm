from typing import Dict, List, Tuple, Optional
import torch

from .memory_manager import MemoryPool


class KVCache:
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
        self.seq_lengths: Dict[int, int] = {}  # seq_id -> current_length

    def allocate(self, seq_id: int, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], seq_length: int):
        seq_length = seq_length - 1
        if not self.memory_pool.allocate_for_sequence(seq_id, seq_length):
            raise RuntimeError(f"Failed to allocate memory for sequence {seq_id}")

        allocations = self.memory_pool.seq_allocations[seq_id]
        # 按块起始位置排序确保序列顺序
        allocations = sorted(allocations, key=lambda x: x[1])  # 按start排序

        # 逐层处理
        for layer_idx, (k, v) in enumerate(past_key_values):
            current_pos = 0  # 当前写入位置
            for block_idx, start, end in allocations:
                chunk_length = end - start  # 当前块的可用长度
                # 切片: [num_heads, chunk_length, head_size]
                k_chunk = k[:, current_pos:current_pos + chunk_length, :]
                v_chunk = v[:, current_pos:current_pos + chunk_length, :]

                block = self.memory_pool.blocks[block_idx]
                # 写入块缓存 (注意维度匹配)
                block.key_cache[layer_idx, :, start:end] = k_chunk
                block.value_cache[layer_idx, :, start:end] = v_chunk

                current_pos += chunk_length  # 更新序列位置

        self.seq_lengths[seq_id] = seq_length

    def delete(self, seq_id: int):
        """删除序列缓存"""
        self.memory_pool.free_sequence(seq_id)
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def get(self, seq_id: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """获取序列的KV缓存"""
        if seq_id not in self.memory_pool.seq_allocations:
            return None

        allocations = self.memory_pool.seq_allocations[seq_id]
        num_layers = self.memory_pool.num_layers
        kv_cache = []

        for layer_idx in range(num_layers):
            ## 形状(num_layers, num_heads, block_size, head_size)
            keys, values = [], []
            for block_idx, start, end in allocations:
                block = self.memory_pool.blocks[block_idx]
                # 按头直接读取连续内存
                keys.append(block.key_cache[layer_idx, :, start:end])  # 形状: (num_heads, chunk_size, head_size)
                values.append(block.value_cache[layer_idx, :, start:end])

            # 沿序列维度拼接
            k = torch.cat(keys, dim=1)  # dim=1拼接seq_len -> (num_heads, total_seq_len, head_size)
            v = torch.cat(values, dim=1)
            kv_cache.append((k, v))

        return kv_cache

    def batch_kv(self, seq_ids: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """批量获取KV缓存"""
        batch_kv = []
        max_len = max(self.seq_lengths.get(seq_id, 0) for seq_id in seq_ids)
        num_layers = self.memory_pool.num_layers

        for layer_idx in range(num_layers):
            keys, values = [], []
            for seq_id in seq_ids:
                k, v = self.get(seq_id)[layer_idx]  # k/v: (num_heads, seq_len, head_size)
                keys.append(k)
                values.append(v)

            # 计算当前层的最大序列长度
            max_len = max(k.size(1) for k in keys)  # 注意：dim=1是seq_len

            padded_keys, padded_values = [], []
            for k, v in zip(keys, values):
                # 创建填充张量 (num_heads, max_len, head_size)
                pad_k = torch.zeros(k.size(0), max_len, k.size(2), device=k.device, dtype=k.dtype)
                pad_k[:, :k.size(1), :] = k  # 复制有效数据
                padded_keys.append(pad_k)

                pad_v = torch.zeros_like(pad_k)
                pad_v[:, :v.size(1), :] = v
                padded_values.append(pad_v)

            batch_k = torch.stack(padded_keys)  # 形状: (batch, num_heads, max_len, head_size)
            batch_v = torch.stack(padded_values)
            batch_kv.append((batch_k, batch_v))

        return batch_kv

    def unbatch_kv(self, seq_ids: List[int], batch_kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[
        int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """拆分批量KV缓存"""
        kv_dict = {}
        for i, seq_id in enumerate(seq_ids):
            seq_kv = []
            for k, v in batch_kv:
                seq_k = k[i:i + 1].squeeze(0)
                seq_v = v[i:i + 1].squeeze(0)
                seq_kv.append((seq_k, seq_v))
            kv_dict[seq_id] = seq_kv
        return kv_dict