# core/scheduler.py
from collections import deque
from typing import List, Tuple

from .sequence import Sequence


class Scheduler:
    def __init__(self, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting_queue = deque()  # 新请求
        self.running_sequences = []   # 正在运行的序列（prefill/decode）
        self.finished_sequences = []  # 已完成

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        batch = []
        total_prefill_tokens = 0
        has_prefill = False
        has_decode = False

        # 1. 优先处理 waiting 队列中的 prefill 请求
        while self.waiting_queue and len(batch) < self.max_batch_size:
            seq = self.waiting_queue[0]
            if total_prefill_tokens + len(seq.input_ids) > self.max_prefill_tokens:
                break
            seq = self.waiting_queue.popleft()
            batch.append(seq)
            total_prefill_tokens += len(seq.input_ids)
            has_prefill = True

        # 2. 如果还有空间，从 running_sequences 中取 decode 请求
        # 注意：只取 state == "decode" 且未完成的
        decode_candidates = [s for s in self.running_sequences if s.state == "decode" and not s.is_finished()]
        while decode_candidates and len(batch) < self.max_batch_size:
            seq = decode_candidates.pop(0)
            batch.append(seq)
            has_decode = True

        # 3. ✅ 关键：把 batch 中的序列从 running_sequences 中移除（如果是 decode）
        for seq in batch:
            if seq in self.running_sequences:
                self.running_sequences.remove(seq)

        # 4. 把 prefill 序列加入 running_sequences（decode 序列已经不在 running 中了）
        for seq in batch:
            if seq.state == "prefill":
                self.running_sequences.append(seq)

        # 5. 判断 batch_type
        if has_prefill and has_decode:
            batch_type = "mixed"
        elif has_prefill:
            batch_type = "prefill"
        elif has_decode:
            batch_type = "decode"
        else:
            return [], "idle"

        return batch, batch_type

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq.seq_id, seq.output_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results
