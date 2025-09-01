# core/scheduler.py (重构)
from collections import deque
import random

from core import Sequence


class Scheduler:
    def __init__(self, max_batch_size=8, max_queue_size=128):
        self.waiting_queue = deque()  # 等待prefill的序列
        self.running_queue = deque()  # 正在运行的序列
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.next_seq_id = 0

    def add_request(self, prompt: str) -> int:
        """添加新请求，返回序列ID"""
        if len(self.waiting_queue) + len(self.running_queue) >= self.max_queue_size:
            return -1  # 队列已满

        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.waiting_queue.append(Sequence(seq_id, prompt))
        return seq_id

    def get_prefill_batch(self) -> list:
        """获取预填充批次"""
        batch = []
        while self.waiting_queue and len(batch) < self.max_batch_size:
            batch.append(self.waiting_queue.popleft())
        return batch

    def get_decode_batch(self) -> list:
        """获取解码批次"""
        batch = []
        # 随机采样保持序列多样性
        candidates = list(self.running_queue)
        random.shuffle(candidates)

        for seq in candidates:
            if len(batch) >= self.max_batch_size:
                break
            if not seq.ended:
                batch.append(seq)

        return batch

    def move_to_running(self, sequences: list):
        """将序列移动到运行队列"""
        for seq in sequences:
            self.running_queue.append(seq)

    def mark_finished(self, seq_id: int):
        """标记序列完成"""
        for seq in self.running_queue:
            if seq.seq_id == seq_id:
                seq.ended = True
                break

    def has_requests(self) -> bool:
        """检查是否有待处理请求"""
        return bool(self.waiting_queue or self.running_queue)