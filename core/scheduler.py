from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from .sequence import Sequence
import logging
import time

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, max_batch_size: int = 32, max_prefill_tokens: int = 2048, 
                 tokenizer: AutoTokenizer = None, prefill_timeout: float = 0.5):
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
        self.batch_sizes = [1, 2, 4, 8, 16, 32]  # 已捕获的 batch_size（与 engine 一致）

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
            logger.info(f"batch: {batch}, batch_type: {batch_type}")
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
                    logger.info(f"batch_len: {batch_len}")
                    batch_sizes = max((b for b in self.batch_sizes if b <= batch_len), default=self.batch_sizes[0])
                    return batch[:batch_sizes], batch_type

        logger.info(f"running_sequences: {len(self.running_sequences)}")
        return [], "idle"

    def _get_prefill_batch(self) -> Tuple[List[Sequence], str]:
        """
        预填充批次调度：基于桶区间的分批策略
        
        策略说明：
        - 将预填充请求按长度映射到 bucket_size 大小的桶中
        - 从最短的桶开始选择批次
        - 同一桶内的请求长度相近，padding 最小化
        
        示例（bucket_size=50）：
        - 长度 [10, 45, 52, 98, 103] 
        - 桶分组: {0: [10, 45], 50: [52], 100: [98, 103]}
        - 优先选择桶 0（最短）
        """
        # 按桶区间分组
        bucket_groups = defaultdict(list)
        
        for seq in list(self.waiting_queue):
            if seq.state == "prefill":
                length = len(seq.input_ids)
                bucket = self._get_bucket_key(length)
                bucket_groups[bucket].append(seq)
        
        if not bucket_groups:
            return [], "idle"
        
        # 从最短的桶开始选择（升序排列）
        sorted_buckets = sorted(bucket_groups.keys())
        
        for bucket_key in sorted_buckets:
            bucket_sequences = bucket_groups[bucket_key]
            # 按长度降序排列（长的在前，减少padding）
            bucket_sequences.sort(key=lambda s: len(s.input_ids), reverse=True)
            
            selected = []
            total_tokens = 0
            for seq in bucket_sequences:
                if len(selected) >= self.max_batch_size:
                    break
                seq_tokens = len(seq.input_ids)
                if total_tokens + seq_tokens > self.max_prefill_tokens:
                    continue
                selected.append(seq)
                total_tokens += seq_tokens
            
            # 触发批次的条件：
            # 1. 达到 max_batch_size
            # 2. 或者桶内最早请求等待时间超过 prefill_timeout（0.1s = 100ms）
            
            # while True:
            #     current_time = time.time()
            #     # 桶内最晚请求的等待时间（当前时间 - 桶内最早到达的请求时间）
            #     min_timestamp = min(seq.timestamp for seq in bucket_sequences)
            #     wait_time = current_time - min_timestamp
                
            if len(selected) >= 32:
                # 找到最长序列的长度
                max_len = max(len(seq.input_ids) for seq in selected)
                
                # 将所有序列填充到最长长度（用 0 填充）
                for seq in selected:
                    current_len = len(seq.input_ids)
                    if current_len < max_len:
                        seq.input_ids = seq.input_ids + [0] * (max_len - current_len)
                
                for seq in selected:
                    self.waiting_queue.remove(seq)
                    self.running_sequences.append(seq)
                
                logger.info(f"prefill bucket: {bucket_key}, selected: {len(selected)} sequences, max_len: {max_len}, tokens: {total_tokens}")
                return selected, "prefill"
                # logger.info(f"not enough sequences, do waiting: {wait_time:.3f}s")
                # time.sleep(0.5)
        
        return [], "idle"

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results
