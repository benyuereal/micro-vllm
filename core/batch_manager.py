# core/batch_manager.py
from collections import deque
from typing import List, Tuple, Dict
import torch


class BatchManager:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.sequences: Dict[int, dict] = {}  # seq_id -> {'tokens': list, 'input_tensor': tensor}
        self.pending_queue = deque()
        self.active_sequences = set()

    def add_sequence(self, seq_id: int, input_tensor: torch.Tensor):
        """添加新序列到批次管理器"""
        self.sequences[seq_id] = {
            'tokens': input_tensor[0].tolist(),
            'input_tensor': input_tensor
        }
        self.pending_queue.append(seq_id)
        self.active_sequences.add(seq_id)

    def update_sequence(self, seq_id: int, input_tensor: torch.Tensor):
        """更新序列的输入张量"""
        if seq_id in self.sequences:
            self.sequences[seq_id]['input_tensor'] = input_tensor
            self.sequences[seq_id]['tokens'].append(input_tensor[0].item())
            self.pending_queue.append(seq_id)

    def finish_sequence(self, seq_id: int):
        """标记序列已完成"""
        if seq_id in self.active_sequences:
            self.active_sequences.remove(seq_id)
        if seq_id in self.sequences:
            # 保留tokens但移除输入张量
            self.sequences[seq_id]['input_tensor'] = None

    def get_batch(self) -> List[Tuple[int, torch.Tensor]]:
        """获取当前批次"""
        batch = []

        # 从队列中获取序列直到达到批次大小或队列为空
        while self.pending_queue and len(batch) < self.max_batch_size:
            seq_id = self.pending_queue.popleft()

            # 只包含活跃序列
            if seq_id in self.active_sequences and seq_id in self.sequences:
                input_tensor = self.sequences[seq_id]['input_tensor']
                if input_tensor is not None:
                    batch.append((seq_id, input_tensor))

        return batch

    def has_active_sequences(self) -> bool:
        """检查是否有活跃序列"""
        return len(self.active_sequences) > 0