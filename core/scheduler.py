from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

from .sequence import Sequence


class Scheduler:
    def __init__(self, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting_queue = deque()   # 新请求
        self.running_sequences = []    # 正在运行的序列
        self.finished_sequences = []   # 已完成

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        batch = []
        batch_type = "idle"

        # 1. 预填充（保持不变）
        if self.waiting_queue:
            length_groups = defaultdict(list)
            for seq in list(self.waiting_queue):
                if seq.state == "prefill":
                    length = len(seq.input_ids)
                    length_groups[length].append(seq)

            if length_groups:
                max_group = max(length_groups.values(), key=len)  # 最大组（同长度最多）
                max_group.sort(key=lambda s: len(s.input_ids), reverse=True)

                selected = []
                total_tokens = 0
                for seq in max_group:
                    if len(selected) >= self.max_batch_size:
                        break
                    seq_tokens = len(seq.input_ids)
                    if total_tokens + seq_tokens > self.max_prefill_tokens:
                        continue
                    selected.append(seq)
                    total_tokens += seq_tokens

                for seq in selected:
                    self.waiting_queue.remove(seq)
                    self.running_sequences.append(seq)

                if selected:
                    batch = selected
                    batch_type = "prefill"

        # 2. 解码：SJF + 同长度成批
        if not batch and self.running_sequences:
            length_groups = defaultdict(list)
            for seq in self.running_sequences:
                if seq.state == "decode" and not seq.is_finished():
                    length = seq.current_position
                    length_groups[length].append(seq)

            if length_groups:
                # 找到最短的长度组（SJF）
                min_length = min(length_groups.keys())
                min_group = length_groups[min_length]

                # 同长度组内可以任意排序（已经是相同长度）
                # 直接取前 max_batch_size 个
                selected = min_group[:self.max_batch_size]

                if selected:
                    batch = selected
                    batch_type = "decode"

        return batch, batch_type

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results
