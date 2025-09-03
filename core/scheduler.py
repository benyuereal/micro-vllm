# core/scheduler.py
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional

from .sequence import Sequence


class Scheduler:
    def __init__(self, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting_queue = deque()  # 新请求
        self.running_sequences = []  # 正在运行的序列（prefill/decode）
        self.finished_sequences = []  # 已完成

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        batch = []
        batch_type = "idle"

        # 1. 优先处理预填充（prefill）请求
        if self.waiting_queue:
            # 按输入长度分组
            length_groups = defaultdict(list)
            for seq in list(self.waiting_queue):
                if seq.state == "prefill":
                    length = len(seq.input_ids)
                    length_groups[length].append(seq)

            if length_groups:
                # 找到最大的组（相同长度的序列最多）
                max_group = max(length_groups.values(), key=len)
                max_group.sort(key=lambda s: len(s.input_ids), reverse=True)  # 按长度降序排序

                # 选择不超过批大小和token限制的序列
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

                # 从等待队列中移除选中的序列
                for seq in selected:
                    self.waiting_queue.remove(seq)
                    self.running_sequences.append(seq)

                if selected:
                    batch = selected
                    batch_type = "prefill"

        # 2. 如果预填充处理完或没有预填充请求，处理解码（decode）请求
        if not batch and self.running_sequences:
            # 按当前解码位置（序列长度）分组
            length_groups = defaultdict(list)
            for seq in self.running_sequences:
                if seq.state == "decode" and not seq.is_finished():
                    length = seq.current_position
                    length_groups[length].append(seq)

            if length_groups:
                # 找到最大的组（相同长度的序列最多）
                max_group = max(length_groups.values(), key=len)
                max_group.sort(key=lambda s: s.current_position, reverse=True)  # 按长度降序排序

                # 选择不超过批大小的序列
                selected = []
                for seq in max_group:
                    if len(selected) >= self.max_batch_size:
                        break
                    selected.append(seq)

                if selected:
                    batch = selected
                    batch_type = "decode"

        return batch, batch_type

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        # 返回完整的生成序列（包含输入）
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results