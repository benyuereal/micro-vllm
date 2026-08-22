from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from .sequence import Sequence
import logging
import time

logger = logging.getLogger(__name__)


class Scheduler:
    """连续批处理调度器（参考 vLLM V2：decode 优先、消除 idle、prefill 立即调度）。

    核心不变量：
    - running 非空时每步必出 decode batch（GPU 永不因等 prefill 而空转）
    - running 空 + waiting 非空时立即 prefill（单条也 prefill，不等攒满/不等超时）
    - 真无工作才返回 idle

    对比旧调度器消除的气泡：
    - 旧：waiting 有请求但没攒够同长度 batch 且没超 prefill_timeout → 返回 idle → GPU 空转
    - 新：waiting 有请求且 running 空 → 立即 prefill；running 非空 → 持续 decode
    """

    def __init__(self, max_batch_size: int = 32, max_prefill_tokens: int = 2048,
                 tokenizer: AutoTokenizer = None, prefill_timeout: float = 0.02,
                 max_chunk_tokens: int = 1024):
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.prefill_timeout = prefill_timeout  # 保留字段（旧接口兼容），新逻辑不再用它攒批
        # chunked prefill：单 chunk 最大 token 数。长 prompt（> max_chunk_tokens）按此切块，
        # 每步 prefill 一块，避免单条长 prompt 长时间独占 prefill step。
        # 默认 1024 = max_position：prompt ≤ 1024 整条 prefill（不切块），与原行为一致且对
        # 不支持 chunked 续写的架构（DeepSeek MLA prefill 用 flash_attn_func 不读 cache 前缀）
        # 安全。Qwen3 prefill 用 flash_attn_with_kvcache 已验证 chunked 续写正确，可显式调小启用。
        self.max_chunk_tokens = max_chunk_tokens
        self.waiting_queue = deque()   # 新请求
        self.running_sequences = []    # 正在 decode 的序列
        self.finished_sequences = []   # 已完成
        # 已捕获的 batch_size（与 engine graph capture 一致）
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        """连续批处理调度。返回 (batch, batch_type)。

        batch_type: "decode" / "prefill" / "idle"
        decode batch 会 pad 到已捕获的 graph batch_size。
        """
        # 1. 剔除已完成的 running
        self.running_sequences = [
            s for s in self.running_sequences
            if not s.is_finished()
        ]

        # 2. 【decode 优先】running 有 decode seq 就持续 decode，绝不让 GPU 等 prefill
        if self.running_sequences:
            batch = self._get_decode_batch()
            if batch:
                return batch, "decode"

        # 3. 无 decode 可做时尝试 prefill：waiting 有新请求，或 running 中有
        #    chunked prefill 进行中的 seq（prefill_done>0 且未完成）需续切。
        has_pending_prefill = (
            bool(self.waiting_queue) or
            any(s.state == "prefill" and s.prefill_remaining > 0 for s in self.running_sequences)
        )
        if has_pending_prefill:
            batch = self._get_prefill_batch()
            if batch:
                return batch, "prefill"

        # 4. 真无工作
        return [], "idle"

    def _get_decode_batch(self) -> List[Sequence]:
        """从 running 选 decode batch：所有 decode seq 进同一 batch（mixed-length），
        pad 到已捕获的 graph batch_size。

        取消同长度分组：decode 走 flash_attn_with_kvcache(block_table, cache_seqlens)，
        cache_seqlens 是 per-seq 的，同批不同长度完全正确（每条独立 seqlen/位置）。
        同长度分组是大 batch 吞吐杀手——变长场景分组后每组仅 1-2 条，decode 退化为串行。
        mixed-length 让所有 decode seq 同步推进，充分利用 batch 算力。
        """
        selected = [s for s in self.running_sequences
                    if s.state == "decode" and not s.is_finished()]
        if not selected:
            return []

        # SJF 排序：短请求优先（先完成释放 slot），但全部进 batch（不再只取一组）
        selected.sort(key=lambda s: s.current_position)
        selected = selected[:self.max_batch_size]
        if not selected:
            return []

        # pad 到已捕获的 graph batch_size（向上取整到 >= len 的最小捕获值）
        batch_len = len(selected)
        batch_size = min((b for b in self.batch_sizes if b >= batch_len),
                         default=self.batch_sizes[-1])
        padded = selected.copy()
        idx = 0
        while len(padded) < batch_size:
            padded.append(padded[idx % len(selected)])  # 循环复制填充（pad seq 不影响真实 seq）
            idx += 1
        return padded

    def _get_prefill_batch(self) -> List[Sequence]:
        """取 prefill batch，支持 chunked prefill。

        seq 归属：chunked 进行中的 seq（state==prefill, prefill_done>0）留在 running，
        由本方法从 running 续切；新 prompt 从 waiting 取。这样 running 同时含 decode seq
        与 prefill-in-progress seq，但 _get_decode_batch 只选 state==decode 的，故 decode
        优先不变；无 decode seq 时 fallthrough 到此续切 prefill。

        两种模式（同一 step 不混，保证 prefill_runner 等长 [B,S] 约束）：
        1) 短 prompt 等长批量：按精确长度分组（prefill 无 attention mask，padding 污染 KV），
           取最短长度组整条 prefill（offset=0, is_last=True）。原行为。
        2) 长 prompt 分块：单条剩余 > max_chunk_tokens 时切一块（batch=1, offset=prefill_done,
           is_last=(chunk_len==remaining)）。切块期间 seq 留在 running，下步优先续切。

        立即 prefill：running 无 decode 时 GPU 本就空转，有请求就 prefill。
        """
        # ---- 优先续切 running 中已开始的 chunked prefill（公平：先做完已开始的）----
        in_progress = [s for s in self.running_sequences
                       if s.state == "prefill" and s.prefill_done > 0 and s.prefill_remaining > 0]
        if in_progress:
            in_progress.sort(key=lambda s: s.timestamp)
            seq = in_progress[0]
            chunk = self._make_chunk(seq)
            self._activate_chunk(seq, chunk)
            logger.info(f"chunked prefill resume seq={seq.seq_id} offset={seq.prefill_done} "
                        f"chunk={chunk[0]} is_last={chunk[1]}")
            return [seq]

        waiting = [s for s in self.waiting_queue if s.state == "prefill"]
        if not waiting:
            return []

        # ---- 短 prompt 等长批量（offset=0, 整条 prefill）----
        length_groups = defaultdict(list)
        for seq in waiting:
            if seq.prefill_remaining <= self.max_chunk_tokens:
                length_groups[seq.prefill_remaining].append(seq)
        for length in sorted(length_groups.keys()):
            group = length_groups[length]
            group.sort(key=lambda s: s.timestamp)  # FIFO

            selected = []
            total_tokens = 0
            for seq in group:
                if len(selected) >= self.max_batch_size:
                    break
                seq_tokens = length
                if total_tokens + seq_tokens > self.max_prefill_tokens:
                    continue
                selected.append(seq)
                total_tokens += seq_tokens

            if not selected:
                continue

            for seq in selected:
                seq._chunk_len = seq.prefill_remaining
                seq._chunk_is_last = True
                self.waiting_queue.remove(seq)
                self.running_sequences.append(seq)
            logger.info(f"prefill len={length}, selected: {len(selected)}, "
                        f"tokens: {total_tokens}, waiting_left: {len(self.waiting_queue)}")
            return selected

        # ---- 长 prompt 开始切块（batch=1）----
        long_seqs = sorted(
            [s for s in waiting if s.prefill_remaining > self.max_chunk_tokens],
            key=lambda s: s.timestamp)
        if long_seqs:
            seq = long_seqs[0]
            chunk = self._make_chunk(seq)
            self._activate_chunk(seq, chunk)
            logger.info(f"chunked prefill start seq={seq.seq_id} total={len(seq.input_ids)} "
                        f"offset={seq.prefill_done} chunk={chunk[0]} is_last={chunk[1]}")
            return [seq]

        return []

    def _make_chunk(self, seq: Sequence):
        """计算 seq 本 step 的 chunk 长度与是否最后一块。"""
        remaining = seq.prefill_remaining
        chunk_len = min(remaining, self.max_chunk_tokens)
        is_last = (chunk_len == remaining)
        return chunk_len, is_last

    def _activate_chunk(self, seq: Sequence, chunk):
        """把 seq 的 chunk 元信息写好。新 prompt 从 waiting 移入 running；
        续切的 seq 已在 running，不重复添加。状态保持 prefill。"""
        chunk_len, is_last = chunk
        seq._chunk_len = chunk_len
        seq._chunk_is_last = is_last
        if seq in self.waiting_queue:
            self.waiting_queue.remove(seq)
        if seq not in self.running_sequences:
            self.running_sequences.append(seq)

    def mark_finished(self, seq: Sequence):
        if seq in self.running_sequences:
            self.running_sequences.remove(seq)
        self.finished_sequences.append(seq)

    def get_finished_results(self):
        results = [(seq, seq.full_ids) for seq in self.finished_sequences]
        self.finished_sequences.clear()
        return results

    def is_finished(self, seq_id: int) -> bool:
        """判断指定序列是否已完成（不在 waiting 和 running 中）。"""
        for seq in self.waiting_queue:
            if seq.seq_id == seq_id:
                return False
        for seq in self.running_sequences:
            if seq.seq_id == seq_id:
                return False
        return True
