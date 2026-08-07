from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from .sequence import Sequence
import logging
import time

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, max_batch_size: int = 32, max_prefill_tokens: int = 2048, 
                 tokenizer: AutoTokenizer = None, prefill_timeout: float = 0.02):
        """
        Args:
            max_batch_size: 最大批次大小
            max_prefill_tokens: 预填充阶段最大 token 数
            tokenizer: 分词器
            prefill_timeout: 预填充阶段超时时间（秒），默认 0.1s
        """
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.prefill_timeout = prefill_timeout
        self.bucket_size = 50  # 预填充长度分桶区间大小
        self.waiting_queue = deque()   # 新请求
        self.running_sequences = []    # 正在运行的序列
        self.finished_sequences = []   # 已完成
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 40]  # 已捕获的 batch_size（与 engine 一致）

    def _get_bucket_key(self, length: int) -> int:
        """
        将长度映射到桶区间
        例如 bucket_size=50:
        - 长度 0-49   → bucket 0
        - 长度 50-99  → bucket 50
        - 长度 100-149 → bucket 100
        """
        return (length // self.bucket_size) * self.bucket_size

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
            # logger.info(f"batch_type: {batch_type}, batch: {len(batch)}, waiting_queue: {len(self.waiting_queue)}, running_sequences: {len(self.running_sequences)}")
            if batch_type != "idle":
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
                    batch_size = min((b for b in self.batch_sizes if b >= batch_len), default=self.batch_sizes[-1])

                    # 【修改2】Padding补齐：简单复制，直到长度达标
                    padded_batch = batch.copy()
                    idx = 0
                    while len(padded_batch) < batch_size:
                        padded_batch.append(padded_batch[idx % len(batch)])  # 循环复制
                        idx += 1

                    return padded_batch, batch_type


        return [], "idle"

    def _get_prefill_batch(self) -> Tuple[List[Sequence], str]:
        """
        预填充批次调度：按【精确长度】分组。

        同一长度组内的请求 input_ids 等长，可直接拼成 [B, S] 定长 batch，
        无需 padding。不同长度的请求分到不同组、不同批次 prefill——
        因为 prefill_layer 用 causal flash_attn 且不带 attention mask，
        任何 padding（如用 0 填充）都会让假 token 参与注意力、污染 KV cache，
        对 DeepSeek（token 0 非 pad）尤其致命，对 Qwen 同样不正确。
        与 decode 阶段按 current_position 分组保持一致。

        触发条件（任一）：
        - 该长度组凑够 max_batch_size 个；
        - 或组内最早请求等待超过 prefill_timeout（避免短请求无限等待）。
        """
        # 按精确 input_ids 长度分组
        length_groups = defaultdict(list)
        for seq in list(self.waiting_queue):
            if seq.state == "prefill":
                length_groups[len(seq.input_ids)].append(seq)

        if not length_groups:
            return [], "idle"

        # 从最短的长度组开始选择（SJF，短请求优先）
        for length in sorted(length_groups.keys()):
            group = length_groups[length]
            # 同长度内按到达时间排序（FIFO）
            group.sort(key=lambda s: s.timestamp)

            selected = []
            total_tokens = 0
            timestamp = None  # 候选中最早到达时间
            for seq in group:
                if len(selected) >= self.max_batch_size:
                    break
                seq_tokens = len(seq.input_ids)
                if total_tokens + seq_tokens > self.max_prefill_tokens:
                    continue
                selected.append(seq)
                total_tokens += seq_tokens
                if timestamp is None or seq.timestamp < timestamp:
                    timestamp = seq.timestamp

            # 触发批次条件：达 max_batch_size 或最早请求等待超时
            if timestamp is None:
                continue
            wait_time = time.time() - timestamp
            if len(selected) >= self.max_batch_size or wait_time >= self.prefill_timeout:
                for seq in selected:
                    self.waiting_queue.remove(seq)
                    self.running_sequences.append(seq)
                logger.info(f"prefill len={length}, selected: {len(selected)} sequences, tokens: {total_tokens}, wait_time: {wait_time:.3f}s")
                return selected, "prefill"

        # 不满足批次条件，返回 waiting 让外部控制等待
        return [], "waiting"

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results

    def is_finished(self, seq_id: int) -> bool:
        """
        判断指定序列是否已完成（不在 waiting 和 running 中）
        
        Args:
            seq_id: 序列 ID
            
        Returns:
            True 表示已完成（要么在 finished_sequences 中，要么已从 waiting/running 中移除）
        """
        # 检查是否在 waiting_queue 中
        for seq in self.waiting_queue:
            if seq.seq_id == seq_id:
                return False
        
        # 检查是否在 running_sequences 中
        for seq in self.running_sequences:
            if seq.seq_id == seq_id:
                return False
        
        # 不在任何队列中，认为已完成
        return True
