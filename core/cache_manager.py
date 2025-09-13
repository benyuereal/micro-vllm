from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
from .memory_manager import MemoryPool


class KVCache:
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
        self.sequence_blocks: Dict[int, List[Tuple[int, int]]] = {}  # seq_id -> [token_key...]
        self.token_to_sequence: Dict[Tuple[int, int], int] = {}  # token_key -> seq_id

    def allocate(self, seq_id: int, tokens: List[Tuple[int, int]], k: torch.Tensor, v: torch.Tensor):
        """
        为序列中的token分配缓存
        tokens: [(token_id, position)] 列表
        k, v: 形状 [num_layers, num_heads, len(tokens), head_size]
        """
        allocated =  []
        if seq_id in self.sequence_blocks:
            ## 如果已经分配过，那么就使用现有的
            allocated = self.sequence_blocks[seq_id]

        for i, (token_id, position) in enumerate(tokens):
            # 尝试查找或分配槽位
            allocation = self.memory_pool.find_token(token_id, position)
            if allocation is None:
                allocation = self.memory_pool.allocate_token(token_id, position)

            block_id, slot_id = allocation
            block = self.memory_pool.get_block(block_id)

            # 更新KV数据
            layer_k = k[:, :, i, :]  # [num_layers, num_heads, head_size]
            layer_v = v[:, :, i, :]
            block.update_slot(slot_id, layer_k, layer_v)

            allocated.append((token_id, position))
            self.token_to_sequence[(token_id, position)] = seq_id

        self.sequence_blocks[seq_id] = allocated
        return allocated

    def get_token_kv(self, token_id: int, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取特定token的KV数据"""
        allocation = self.memory_pool.find_token(token_id, position)
        if allocation is None:
            raise ValueError(f"Token {(token_id, position)} not found in cache")

        block_id, slot_id = allocation
        block = self.memory_pool.get_block(block_id)
        return block.get_slot(slot_id)



    def delete(self, seq_id: int):
        """删除序列缓存"""
        if seq_id not in self.sequence_blocks:
            return

        for token_key in self.sequence_blocks[seq_id]:
            token_id, position = token_key
            self.memory_pool.free_token(token_id, position)
            if token_key in self.token_to_sequence:
                del self.token_to_sequence[token_key]

        del self.sequence_blocks[seq_id]

    def get_sequence_tokens(self, seq_id: int) -> List[Tuple[int, int]]:
        """获取序列的所有token标识"""
        return self.sequence_blocks.get(seq_id, [])


    def get_block_table(self) -> Dict[int, List[Tuple[int, int]]]:
        """获取块表信息"""
        block_table = defaultdict(list)
        for token_key in self.token_to_sequence:
            token_id, position = token_key
            allocation = self.memory_pool.find_token(token_id, position)
            if allocation:
                block_id, slot_id = allocation
                block_table[block_id].append((token_id, position, slot_id))
        return dict(block_table)