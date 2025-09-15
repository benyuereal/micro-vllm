import time

import torch
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import heapq


class MemoryBlock:
    def __init__(self,
                 block_id: int,
                 block_size: int,
                 num_layers: int,
                 num_heads: int,
                 head_size: int,
                 dtype=torch.float16):
        self.block_id = block_id
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        # 自动选择设备
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda'

        # KV缓存存储 [layers, heads, slots, head_size]
        self.key_cache = torch.zeros(
            (num_layers, num_heads, block_size, head_size),
            dtype=dtype, device=self.device
        )
        self.value_cache = torch.zeros_like(self.key_cache)

        # Token到槽位的映射
        self.token_to_slots: Dict[Tuple[int, int], Set[int]] = defaultdict(set)  # (token_id, position) -> slot_ids
        self.slot_to_token: Dict[int, Tuple[int, int]] = {}  # slot_id -> (token_id, position)
        self.free_slots: deque = deque(range(block_size))
        self.ref_count = 0
        self.last_used = 0  # 记录最后使用时间

    def allocate_slot(self, token_id: int, position: int) -> int:
        """为token分配槽位"""
        if not self.free_slots:
            return -1

        slot_id = self.free_slots.popleft()
        self.slot_to_token[slot_id] = (token_id, position)
        self.token_to_slots[(token_id, position)].add(slot_id)
        self.ref_count += 1
        self.last_used = time.time()
        return slot_id

    def free_slot(self, slot_id: int):
        """释放槽位"""
        if slot_id not in self.slot_to_token:
            return

        token_info = self.slot_to_token.pop(slot_id)
        self.token_to_slots[token_info].discard(slot_id)
        if not self.token_to_slots[token_info]:
            del self.token_to_slots[token_info]

        self.free_slots.append(slot_id)
        self.ref_count -= 1
        self.last_used = time.time()

    def get_slot(self, slot_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取槽位的KV数据"""
        if slot_id not in self.slot_to_token:
            raise ValueError(f"Slot {slot_id} not allocated")
        return (
            self.key_cache[:, :, slot_id, :],
            self.value_cache[:, :, slot_id, :]
        )

    def update_slot(self, slot_id: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """更新槽位的KV数据"""
        if slot_id not in self.slot_to_token:
            raise ValueError(f"Slot {slot_id} not allocated")
        self.key_cache[:, :, slot_id, :].copy_(new_k)
        self.value_cache[:, :, slot_id, :].copy_(new_v)
        self.last_used = time.time()

    def get_free_slot_count(self) -> int:
        """获取空闲槽位数量"""
        return len(self.free_slots)

    def is_empty(self) -> bool:
        """检查块是否为空"""
        return self.ref_count == 0


class MemoryPool:
    def __init__(self,
                 block_size: int = 16,
                 max_blocks: int = 2048,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 head_size: int = 128,
                 dtype=torch.float16):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        self.blocks: List[MemoryBlock] = []
        self.token_map: Dict[Tuple[int, int], List[int]] = {}  # (token_id, position) -> [block_id, slot_id]
        self.token_ref_count: Dict[Tuple[int, int], int] = defaultdict(int)

        # 使用最小堆来管理空闲块，按最后使用时间排序
        self.free_blocks_heap = []  # (last_used, block_id)
        self.free_blocks_set = set()  # 快速查找集合

        # 创建初始块
        for i in range(min(4, max_blocks)):
            self._create_block()

    def _create_block(self) -> MemoryBlock:
        """创建新的内存块"""
        if len(self.blocks) >= self.max_blocks:
            raise RuntimeError("Exceeded maximum blocks limit")

        new_block = MemoryBlock(
            block_id=len(self.blocks),
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            dtype=self.dtype
        )
        self.blocks.append(new_block)
        self._add_to_free_blocks(new_block.block_id)
        return new_block

    def _add_to_free_blocks(self, block_id: int):
        """将块添加到空闲块管理"""
        block = self.blocks[block_id]
        heapq.heappush(self.free_blocks_heap, (block.last_used, block_id))
        self.free_blocks_set.add(block_id)

    def _remove_from_free_blocks(self, block_id: int):
        """从空闲块管理中移除块"""
        self.free_blocks_set.discard(block_id)
        # 注意：堆中的元素不会立即移除，而是在弹出时检查有效性

    def _get_best_free_block(self) -> Optional[int]:
        """获取最佳的空闲块（最近最少使用的）"""
        while self.free_blocks_heap:
            last_used, block_id = heapq.heappop(self.free_blocks_heap)
            if block_id in self.free_blocks_set and self.blocks[block_id].get_free_slot_count() > 0:
                return block_id
        return None

    def find_token(self, token_id: int, position: int) -> Optional[Tuple[int, int]]:
        """查找token是否已经存在"""
        token_key = (token_id, position)
        if token_key in self.token_map:
            return self.token_map[token_key]
        return None

    def allocate_token(self, token_id: int, position: int) -> Tuple[int, int]:
        """
        为token分配槽位
        返回 (block_id, slot_id)
        """
        token_key = (token_id, position)

        # 检查token是否已存在
        if token_key in self.token_map:
            self.token_ref_count[token_key] += 1
            return self.token_map[token_key]

        # 尝试在现有块中分配
        block_id = self._get_best_free_block()
        if block_id is not None:
            block = self.blocks[block_id]
            slot_id = block.allocate_slot(token_id, position)
            if slot_id != -1:
                self.token_map[token_key] = (block_id, slot_id)
                self.token_ref_count[token_key] = 1
                if block.get_free_slot_count() == 0:
                    self._remove_from_free_blocks(block_id)
                return (block_id, slot_id)

        # 创建新块
        if len(self.blocks) < self.max_blocks:
            new_block = self._create_block()
            block_id = new_block.block_id
            slot_id = new_block.allocate_slot(token_id, position)
            if slot_id != -1:
                self.token_map[token_key] = (block_id, slot_id)
                self.token_ref_count[token_key] = 1
                if new_block.get_free_slot_count() < self.block_size:
                    self._add_to_free_blocks(block_id)
                return (block_id, slot_id)

        # 如果所有块都满了，尝试清理并重用最旧的块
        if self.blocks:
            print("block is full")
            # 找到引用计数最少的块
            oldest_block_id = min(
                range(len(self.blocks)),
                key=lambda i: (self.blocks[i].last_used, self.blocks[i].ref_count)
            )
            oldest_block = self.blocks[oldest_block_id]

            # 清理块中的所有token
            slots_to_free = list(oldest_block.slot_to_token.keys())
            for slot_id in slots_to_free:
                token_info = oldest_block.slot_to_token[slot_id]
                self._free_token_from_block(token_info[0], token_info[1], oldest_block_id, slot_id)

            # 重新分配
            slot_id = oldest_block.allocate_slot(token_id, position)
            if slot_id != -1:
                self.token_map[token_key] = (oldest_block_id, slot_id)
                self.token_ref_count[token_key] = 1
                self._add_to_free_blocks(oldest_block_id)
                return (oldest_block_id, slot_id)

        raise RuntimeError("Failed to allocate token slot")

    def _free_token_from_block(self, token_id: int, position: int, block_id: int, slot_id: int):
        """从特定块中释放token"""
        token_key = (token_id, position)

        if token_key in self.token_ref_count:
            self.token_ref_count[token_key] -= 1
            if self.token_ref_count[token_key] <= 0:
                del self.token_ref_count[token_key]
                if token_key in self.token_map:
                    del self.token_map[token_key]

        block = self.blocks[block_id]
        block.free_slot(slot_id)
        if block.get_free_slot_count() > 0:
            self._add_to_free_blocks(block_id)

    def free_token(self, token_id: int, position: int):
        """释放token占用的槽位"""
        token_key = (token_id, position)
        if token_key not in self.token_map:
            return

        self.token_ref_count[token_key] -= 1
        if self.token_ref_count[token_key] > 0:
            return

        block_id, slot_id = self.token_map[token_key]
        if block_id < len(self.blocks):
            block = self.blocks[block_id]
            block.free_slot(slot_id)
            if block.get_free_slot_count() > 0:
                self._add_to_free_blocks(block_id)

        del self.token_map[token_key]
        del self.token_ref_count[token_key]

    def get_block(self, block_id: int) -> MemoryBlock:
        """获取指定块"""
        return self.blocks[block_id]

    def get_all_blocks(self) -> List[MemoryBlock]:
        """获取所有内存块"""
        return self.blocks

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        total_blocks = len(self.blocks)
        used_blocks = total_blocks - len(self.free_blocks_set)
        total_slots = total_blocks * self.block_size
        used_slots = sum(block.ref_count for block in self.blocks)

        # 计算内存占用（MB）
        block_memory = self.block_size * self.num_layers * self.num_heads * self.head_size * 2  # 2 for k and v
        block_memory_mb = block_memory * torch.finfo(self.dtype).bits / 8 / 1024 / 1024

        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "total_slots": total_slots,
            "used_slots": used_slots,
            "memory_used_mb": used_blocks * block_memory_mb,
            "memory_total_mb": total_blocks * block_memory_mb
        }