from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from .sequence import Sequence


class Scheduler:
    def __init__(self, max_batch_size: int = 32, max_prefill_tokens: int = 2048, tokenizer: AutoTokenizer = None):
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting_queue = deque()   # 新请求
        self.running_sequences = []    # 正在运行的序列
        self.finished_sequences = []   # 已完成
        self.batch_sizes = [1, 2, 4, 8, 16, 32]  # 已捕获的 batch_size（与 engine 一致）

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        """
        连续批处理：Padding 凑齐 batch + 动态剔除完成
        核心逻辑：
        1. 剔除已完成的请求
        2. 有新请求时处理 prefill
        3. 解码阶段 Padding 填充到已捕获的 batch_size
        """
        # 1. 剔除已完成的请求
        self.running_sequences = [
            s for s in self.running_sequences
            if not s.is_finished()
        ]

        # 2. 预填充阶段：有新请求时优先处理
        if self.waiting_queue:
            batch, batch_type = self._get_prefill_batch()
            if batch:
                return batch, batch_type

        # 3. 解码阶段：Padding 凑齐 batch_size
        if self.running_sequences:
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
                    # 从 batch_sizes 中找到第一个 <= len(batch) 的值（向下取整）
                    batch_len = len(batch)
                    batch_sizes = max((b for b in self.batch_sizes if b <= batch_len), default=self.batch_sizes[0])
                    return batch[:batch_sizes], batch_type

        return [], "idle"

    def _get_prefill_batch(self) -> Tuple[List[Sequence], str]:
        """提取 prefill 逻辑，复用给连续批处理"""
        batch = []
        length_groups = defaultdict(list)

        for seq in list(self.waiting_queue):
            if seq.state == "prefill":
                length = len(seq.input_ids)
                length_groups[length].append(seq)

        if not length_groups:
            return [], "idle"

        max_group = max(length_groups.values(), key=len)
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
            return selected, "prefill"

        return [], "idle"

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results
