import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class Page:
    """表示一个KV缓存页"""

    def __init__(self, page_id: int, page_size: int, num_heads: int, head_size: int, device: str = "cpu"):
        self.page_id = page_id
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device

        self.k_data = torch.zeros(page_size, num_heads, head_size, device=device)
        self.v_data = torch.zeros(page_size, num_heads, head_size, device=device)
        self.used_slots = 0
        self.is_dirty = False

    def add_entry(self, k_entry: torch.Tensor, v_entry: torch.Tensor) -> int:
        if self.used_slots >= self.page_size:
            raise ValueError("Page is full")

        slot_idx = self.used_slots
        self.k_data[slot_idx] = k_entry
        self.v_data[slot_idx] = v_entry
        self.used_slots += 1
        self.is_dirty = True
        return slot_idx

    def get_entries(self, start_slot: int, end_slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_data[start_slot:end_slot], self.v_data[start_slot:end_slot]


class PagedKVCache:
    """分页KV缓存管理器"""

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 num_key_value_heads: int,
                 head_size: int,
                 page_size: int = 256,
                 max_num_seqs: int = 256,
                 memory_manager=None,
                 device: str = "cuda",
                 max_seq_length: int = 2048):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.page_size = page_size
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.max_seq_length = max_seq_length
        self.num_key_value_heads = num_key_value_heads

        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device

        self.page_pool = []
        self.free_pages = []
        self.sequence_table = {}
        self.sequence_lengths = {}
        self._init_pages()

    def _init_pages(self):
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
        if self.free_pages:
            return self.free_pages.pop()

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
        page = self.page_pool[page_id]
        page.used_slots = 0
        page.is_dirty = False
        self.free_pages.append(page_id)

    def add_sequence(self, sequence_id: int):
        if sequence_id in self.sequence_table:
            return

        self.sequence_table[sequence_id] = []
        self.sequence_lengths[sequence_id] = 0

    def remove_sequence(self, sequence_id: int):
        if sequence_id not in self.sequence_table:
            return

        for page_id in self.sequence_table[sequence_id]:
            self.free_page(page_id)

        del self.sequence_table[sequence_id]
        del self.sequence_lengths[sequence_id]

    def get_sequence_length(self, sequence_id: int) -> int:
        return self.sequence_lengths.get(sequence_id, 0)

    def update_cache(self,
                     hidden_states: torch.Tensor,
                     sequence_ids: List[int],
                     positions: torch.Tensor):

        batch_size, seq_len, _ = hidden_states.shape

        for i in range(batch_size):
            seq_id = sequence_ids[i]

            if seq_id not in self.sequence_table:
                self.add_sequence(seq_id)

            pages = self.sequence_table[seq_id]
            current_length = self.sequence_lengths[seq_id]

            total_slots_needed = current_length + seq_len
            total_pages_needed = (total_slots_needed + self.page_size - 1) // self.page_size
            new_pages_needed = max(0, total_pages_needed - len(pages))

            if new_pages_needed > 0 or len(pages) == 0:
                for _ in range(new_pages_needed):
                    new_page_id = self.allocate_page()
                    pages.append(new_page_id)
                if len(pages) == 0:
                    new_page_id = self.allocate_page()
                    pages.append(new_page_id)

            for j in range(seq_len):
                pos = current_length + j
                page_idx = pos // self.page_size
                slot_idx = pos % self.page_size

                if page_idx >= len(pages):
                    new_page_id = self.allocate_page()
                    pages.append(new_page_id)
                    page_idx = len(pages) - 1

                page_id = pages[page_idx]
                page = self.page_pool[page_id]

                k_data = hidden_states[i, j].reshape(self.num_heads, self.head_size)
                v_data = hidden_states[i, j].reshape(self.num_heads, self.head_size)

                if slot_idx >= page.used_slots:
                    page.add_entry(k_data, v_data)

            self.sequence_lengths[seq_id] = current_length + seq_len

    def get_cache(self,
                  sequence_ids: List[int],
                  past_seq_lengths: List[int]) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:

        batch_size = len(sequence_ids)
        max_actual_length = max(past_seq_lengths) if batch_size > 0 else 0
        if max_actual_length == 0:
            return None

        max_length = min(max_actual_length, self.max_seq_length)
        past_key_values = []

        for layer_idx in range(self.num_layers):
            k_list = []
            v_list = []

            for i in range(batch_size):
                seq_id = sequence_ids[i]
                seq_length = min(past_seq_lengths[i], self.max_seq_length)

                if seq_id not in self.sequence_table or seq_length == 0:
                    k_tensor = torch.zeros(
                        self.num_key_value_heads, max_length, self.head_size,
                        device=self.device
                    )
                    v_tensor = torch.zeros(
                        self.num_key_value_heads, max_length, self.head_size,
                        device=self.device
                    )
                else:
                    pages = self.sequence_table[seq_id]
                    start_page = 0
                    end_page = (seq_length - 1) // self.page_size + 1

                    k_slices = []
                    v_slices = []
                    for page_idx in range(start_page, end_page):
                        page_id = pages[page_idx]
                        page = self.page_pool[page_id]

                        start_slot = 0
                        end_slot = self.page_size
                        if page_idx == end_page - 1:
                            end_slot = seq_length % self.page_size or self.page_size

                        k_slice, v_slice = page.get_entries(start_slot, end_slot)
                        k_slice = k_slice.permute(1, 0, 2)
                        v_slice = v_slice.permute(1, 0, 2)

                        k_slices.append(k_slice)
                        v_slices.append(v_slice)

                    if k_slices:
                        k_tensor = torch.cat(k_slices, dim=1)
                        v_tensor = torch.cat(v_slices, dim=1)
                    else:
                        k_tensor = torch.zeros(
                            self.num_key_value_heads, max_length, self.head_size,
                            device=self.device
                        )
                        v_tensor = torch.zeros(
                            self.num_key_value_heads, max_length, self.head_size,
                            device=self.device
                        )

                    if k_tensor.size(1) < max_length:
                        padding_size = max_length - k_tensor.size(1)
                        k_tensor = F.pad(k_tensor, (0, 0, 0, padding_size))
                        v_tensor = F.pad(v_tensor, (0, 0, 0, padding_size))

                k_list.append(k_tensor)
                v_list.append(v_tensor)

            k_batch = torch.stack(k_list)
            v_batch = torch.stack(v_list)
            past_key_values.append((k_batch, v_batch))

        return past_key_values