import torch
from typing import Dict, List, Tuple, Optional
import math


class MemoryBlock:
    def __init__(self, block_size: int, num_layers: int, head_size: int, num_heads: int, dtype=torch.float16):
        self.block_size = block_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.num_heads = num_heads

        # 预分配内存块
        self.key_cache = torch.zeros(
            (num_layers, num_heads, block_size, head_size),
            dtype=dtype, device='mps'
        )
        self.value_cache = torch.zeros_like(self.key_cache)

        # 使用情况跟踪
        self.used = 0
        self.allocated_indices = set()

    def allocate(self, num_tokens: int) -> Tuple[int, int]:
        """分配连续空间，返回(start, end)索引"""
        if self.used + num_tokens > self.block_size:
            return (-1, -1)

        start = self.used
        end = start + num_tokens
        self.used += num_tokens

        for i in range(start, end):
            self.allocated_indices.add(i)

        return (start, end)

    def free(self, start: int, end: int):
        """释放已分配的空间"""
        for i in range(start, end):
            if i in self.allocated_indices:
                self.allocated_indices.remove(i)
        # 可以在这里实现内存压缩逻辑


class MemoryPool:
    def __init__(self,
                 block_size: int = 1024,
                 max_blocks: int = 32,
                 num_layers: int = 32,
                 head_size: int = 128,
                 num_heads: int = 32,
                 dtype=torch.float32):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_layers = num_layers
        self.head_size = head_size
        self.num_heads = num_heads
        self.dtype = dtype  # 保存数据类型

        self.blocks: List[MemoryBlock] = []
        self.seq_allocations: Dict[int, List[Tuple[int, int, int]]] = {}  # seq_id -> [(block_idx, start, end)]

    def allocate_for_sequence(self, seq_id: int, seq_length: int) -> bool:
        """为序列分配内存"""
        if seq_id in self.seq_allocations:
            return True

        needed_blocks = math.ceil(seq_length / self.block_size)
        allocated = []

        # 尝试在现有块中分配
        for block_idx, block in enumerate(self.blocks):
            if len(allocated) >= needed_blocks:
                break

            start, end = block.allocate(seq_length)
            if start != -1:
                allocated.append((block_idx, start, end))

        # 如果现有块不足，创建新块
        while len(allocated) < needed_blocks and len(self.blocks) < self.max_blocks:
            new_block = MemoryBlock(
                self.block_size, self.num_layers,
                self.head_size, self.num_heads, self.dtype
            )
            self.blocks.append(new_block)
            block_idx = len(self.blocks) - 1

            start, end = new_block.allocate(seq_length)
            if start != -1:
                allocated.append((block_idx, start, end))

        if len(allocated) >= needed_blocks:
            self.seq_allocations[seq_id] = allocated
            return True
        return False

    def free_sequence(self, seq_id: int):
        """释放序列占用的内存"""
        if seq_id not in self.seq_allocations:
            return

        for block_idx, start, end in self.seq_allocations[seq_id]:
            if block_idx < len(self.blocks):
                self.blocks[block_idx].free(start, end)

        del self.seq_allocations[seq_id]