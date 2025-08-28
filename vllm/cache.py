import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class Page:
    """表示一个KV缓存页"""

    def __init__(self, page_id: int, page_size: int, num_heads: int, head_size: int, device: str = "cpu"):
        self.page_id = page_id
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device

        # 初始化键值缓存
        self.k_data = torch.zeros(page_size, num_heads, head_size, device=device)
        self.v_data = torch.zeros(page_size, num_heads, head_size, device=device)

        # 页面使用情况
        self.used_slots = 0
        self.is_dirty = False

    def add_entry(self, k_entry: torch.Tensor, v_entry: torch.Tensor) -> int:
        """向页面添加键值对"""
        if self.used_slots >= self.page_size:
            raise ValueError("Page is full")

        slot_idx = self.used_slots
        self.k_data[slot_idx] = k_entry
        self.v_data[slot_idx] = v_entry

        self.used_slots += 1
        self.is_dirty = True

        return slot_idx

    def get_entries(self, start_slot: int, end_slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定范围的键值对"""
        return self.k_data[start_slot:end_slot].to(self.device), self.v_data[start_slot:end_slot].to(self.device)


class PagedKVCache:
    """分页KV缓存管理器"""

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 head_size: int,
                 page_size: int = 256,
                 max_num_seqs: int = 256,
                 memory_manager=None,
                 device: str = "cuda"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.page_size = page_size
        self.max_num_seqs = max_num_seqs
        self.memory_manager = memory_manager
        self.device = device
        # 设备自动补全
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device

        # 初始化页面池
        self.page_pool = []
        self.free_pages = []

        # 序列管理
        self.sequence_table = {}  # sequence_id -> List[Page]
        self.sequence_lengths = {}  # sequence_id -> length

        # 初始化页面
        self._init_pages()

    def _init_pages(self):
        """初始化页面池"""
        # 初始页面数量基于最大序列数
        initial_num_pages = self.max_num_seqs * 4

        for i in range(initial_num_pages):
            page = Page(
                page_id=i,
                page_size=self.page_size,
                num_heads=self.num_heads,
                head_size=self.head_size,
                device=self.device
            )
            self.page_pool.append(page)
            self.free_pages.append(i)

    def allocate_page(self) -> int:
        """分配一个新页面"""
        if self.free_pages:
            return self.free_pages.pop()

        # 如果没有空闲页面，创建一个新页面
        new_page_id = len(self.page_pool)
        new_page = Page(
            page_id=new_page_id,
            page_size=self.page_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
            device=self.device
        )

        self.page_pool.append(new_page)
        return new_page_id

    def free_page(self, page_id: int):
        """释放页面"""
        page = self.page_pool[page_id]
        page.used_slots = 0
        page.is_dirty = False
        self.free_pages.append(page_id)

    def add_sequence(self, sequence_id: int):
        """添加一个新序列"""
        if sequence_id in self.sequence_table:
            return

        self.sequence_table[sequence_id] = []
        self.sequence_lengths[sequence_id] = 0

    def remove_sequence(self, sequence_id: int):
        """移除一个序列"""
        if sequence_id not in self.sequence_table:
            return

        # 释放所有页面
        for page_id in self.sequence_table[sequence_id]:
            self.free_page(page_id)

        # 从表中移除
        del self.sequence_table[sequence_id]
        del self.sequence_lengths[sequence_id]

    def get_sequence_length(self, sequence_id: int) -> int:
        """获取序列长度"""
        return self.sequence_lengths.get(sequence_id, 0)

    def update_cache(self,
                     hidden_states: torch.Tensor,
                     sequence_ids: List[int],
                     positions: torch.Tensor):
        """更新KV缓存"""
        batch_size, seq_len, _ = hidden_states.shape

        for i in range(batch_size):
            seq_id = sequence_ids[i]

            # 确保序列存在
            if seq_id not in self.sequence_table:
                self.add_sequence(seq_id)

            # 获取当前序列的页面
            pages = self.sequence_table[seq_id]
            current_length = self.sequence_lengths[seq_id]

            # 计算需要的新页面数
            needed_slots = seq_len
            current_slots = current_length % self.page_size
            needed_pages = (current_slots + needed_slots - 1) // self.page_size

            # 分配新页面
            for _ in range(needed_pages - len(pages)):
                new_page_id = self.allocate_page()
                pages.append(new_page_id)

            # 更新缓存
            for j in range(seq_len):
                pos = current_length + j
                page_idx = pos // self.page_size
                slot_idx = pos % self.page_size

                page_id = pages[page_idx]
                page = self.page_pool[page_id]

                # 获取当前层的键值（简化实现）
                k_data = hidden_states[i, j, :].view(self.num_heads, self.head_size)
                v_data = hidden_states[i, j, :].view(self.num_heads, self.head_size)

                # 添加到页面
                if slot_idx >= page.used_slots:
                    page.add_entry(k_data, v_data)

            # 更新序列长度
            self.sequence_lengths[seq_id] = current_length + seq_len

    def get_cache(self,
                  sequence_ids: List[int],
                  past_seq_lengths: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """获取KV缓存，返回格式为每层一个元组(k_tensor, v_tensor)"""
        batch_size = len(sequence_ids)
        past_key_values = []

        for layer_idx in range(self.num_layers):
            k_list = []
            v_list = []

            for i in range(batch_size):
                seq_id = sequence_ids[i]
                seq_length = past_seq_lengths[i]

                if seq_id not in self.sequence_table or seq_length == 0:
                    # 创建空张量
                    k_list.append(torch.zeros(0, self.num_heads, self.head_size, device=self.device))
                    v_list.append(torch.zeros(0, self.num_heads, self.head_size, device=self.device))
                    continue

                # 获取序列的所有页面
                pages = self.sequence_table[seq_id]
                start_page = 0
                end_page = (seq_length - 1) // self.page_size + 1

                k_slices = []
                v_slices = []
                for page_idx in range(start_page, end_page):
                    page_id = pages[page_idx]
                    page = self.page_pool[page_id]

                    # 计算页面内的槽位
                    start_slot = 0
                    end_slot = self.page_size
                    if page_idx == end_page - 1:  # 最后一页
                        end_slot = seq_length % self.page_size or self.page_size

                    # 获取数据
                    k_slice, v_slice = page.get_entries(start_slot, end_slot)
                    k_slices.append(k_slice)
                    v_slices.append(v_slice)

                # 拼接所有切片
                if k_slices:
                    k_tensor = torch.cat(k_slices, dim=0)
                    v_tensor = torch.cat(v_slices, dim=0)
                else:
                    k_tensor = torch.zeros(0, self.num_heads, self.head_size, device=self.device)
                    v_tensor = torch.zeros(0, self.num_heads, self.head_size, device=self.device)

                k_list.append(k_tensor)
                v_list.append(v_tensor)

            # 将列表转换为批次张量 [batch_size, seq_len, num_heads, head_size]
            k_batch = torch.stack(k_list)  # [batch, seq_len, num_heads, head_size]
            v_batch = torch.stack(v_list)

            # 调整维度顺序 [batch, num_heads, seq_len, head_size]
            k_batch = k_batch.permute(0, 2, 1, 3)
            v_batch = v_batch.permute(0, 2, 1, 3)

            # 将当前层的缓存添加到结果中
            past_key_values.append((k_batch, v_batch))

        return past_key_values