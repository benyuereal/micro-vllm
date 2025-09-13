import torch
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


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
        self.free_slots: List[int] = list(range(block_size))
        self.ref_count = 0

    def allocate_slot(self, token_id: int, position: int) -> int:
        """为token分配槽位"""
        if not self.free_slots:
            return -1

        slot_id = self.free_slots.pop()
        self.slot_to_token[slot_id] = (token_id, position)
        self.token_to_slots[(token_id, position)].add(slot_id)
        self.ref_count += 1
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
        self.key_cache[:, :, slot_id, :] = new_k
        self.value_cache[:, :, slot_id, :] = new_v

    def get_free_slot_count(self) -> int:
        """获取空闲槽位数量"""
        return len(self.free_slots)


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
        self.free_blocks = set()

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
        self.free_blocks.add(new_block.block_id)
        return new_block

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
        for block_id in list(self.free_blocks):
            block = self.blocks[block_id]
            if block.get_free_slot_count() > 0:
                slot_id = block.allocate_slot(token_id, position)
                if slot_id != -1:
                    self.token_map[token_key] = (block_id, slot_id)
                    self.token_ref_count[token_key] = 1
                    if block.get_free_slot_count() == 0:
                        self.free_blocks.discard(block_id)
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
                    self.free_blocks.add(block_id)
                return (block_id, slot_id)

        raise RuntimeError("Failed to allocate token slot")

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
                self.free_blocks.add(block_id)

        del self.token_map[token_key]
        del self.token_ref_count[token_key]

    def get_block(self, block_id: int) -> MemoryBlock:
        """获取指定块"""
        return self.blocks[block_id]

    def get_all_blocks(self) -> List[MemoryBlock]:
        """获取所有内存块"""
        return self.blocks