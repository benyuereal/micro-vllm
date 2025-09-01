# core/scheduler.py
from collections import deque
from typing import List, Dict, Tuple

class Scheduler:
    def __init__(self, max_batch_size=8):
        self.prefill_queue = deque()  # 新请求队列
        self.decode_queue = deque()   # 解码中请求队列
        self.max_batch_size = max_batch_size
        self.sequence_states: Dict[int, Dict] = {}  # 序列ID -> 状态元数据

    def add_request(self, prompt: str, max_tokens: int):
        seq_id = id(prompt)
        self.prefill_queue.append({
            "seq_id": seq_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "generated_length": 0
        })
        self.sequence_states[seq_id] = {"status": "waiting"}

    def get_batch(self) -> Tuple[List, List]:
        """动态混合新请求和解码请求"""
        prefill_batch = []
        while self.prefill_queue and len(prefill_batch) < self.max_batch_size:
            prefill_batch.append(self.prefill_queue.popleft())

        decode_batch = []
        active_sequences = [s for s in self.decode_queue if self.sequence_states[s["seq_id"]]["status"] == "decoding"]
        decode_batch = active_sequences[:self.max_batch_size - len(prefill_batch)]

        return prefill_batch, decode_batch

    def update_sequence_state(self, seq_id: int, **kwargs):
        if seq_id in self.sequence_states:
            self.sequence_states[seq_id].update(kwargs)
            if kwargs.get("status") == "decoding":
                self.decode_queue.append({"seq_id": seq_id})