from collections import deque
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from .sequence import Sequence
import logging
import time

logger = logging.getLogger(__name__)


class Scheduler:
    """连续批处理调度器（prefill 优先，借鉴 nano-vllm：快速填满 running 再 decode）。

    核心不变量：
    - waiting 非空（或有 chunked prefill 在途）时优先 prefill，尽快把请求推进 running，
      使 decode batch 尽快长到 max_batch_size —— prefill 被饿死会导致 batch 长不大、吞吐低。
    - 无 prefill 可做时 decode（最短序列优先 SJF：短序列先完成释放 slot）。
    - prefill 选不出 batch（如 KV block 不足）时 fallthrough 到 decode，decode 释放 block 后
      下步再 prefill —— 兼具 OOM 自愈。
    - 真无工作才返回 idle。

    对比 decode-优先策略：decode-优先下 running 一旦非空就持续 decode，prefill 被推迟到
    running 清空才执行，running 在 admission 与 completion 间稳态平衡，batch 远小于
    max_batch_size；prefill-优先下连续 prefill 把 running 推满，decode batch 接近上限。
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
        # 已捕获的 batch_size（engine 初始化后会被覆写为真实 cap_sizes，保持单一来源）。
        # 默认值仅作 fallback：小 bs 密集，大 bs 按 1.5x（padding 率 ≤1.33x）。
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

    def add_request(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        """连续批处理调度（prefill 优先，借鉴 nano-vllm）。返回 (batch, batch_type)。

        batch_type: "decode" / "prefill" / "idle"
        - waiting 非空（或有 chunked prefill 在途）时优先 prefill，尽快把 running 推满；
        - 无 prefill 可做时 decode（SJF 最短序列优先）；
        - prefill 选不出（如 KV block 不足）时 fallthrough 到 decode，decode 释放 block 后下步再 prefill。
        decode batch 会 pad 到已捕获的 graph batch_size。
        """
        # 1. 剔除已完成的 running
        self.running_sequences = [
            s for s in self.running_sequences
            if not s.is_finished()
        ]

        # 2. 【prefill 优先】有新请求或在途 chunked prefill 就先 prefill，快速填满 running
        has_pending_prefill = (
            bool(self.waiting_queue) or
            any(s.state == "prefill" and s.prefill_remaining > 0 for s in self.running_sequences)
        )
        if has_pending_prefill:
            batch = self._get_prefill_batch()
            if batch:
                return batch, "prefill"

        # 3. 无 prefill 可做（waiting 空 / block 不足选不出）时 decode
        batch = self._get_decode_batch()
        if batch:
            return batch, "decode"

        # 4. 真无工作
        return [], "idle"

    def _get_decode_batch(self) -> List[Sequence]:
        """从 running 选 decode batch：所有 decode seq 进同一 batch（mixed-length），
        pad 到已捕获的 graph batch_size。

        decode 走 flash_attn_with_kvcache(block_table, cache_seqlens)，cache_seqlens 是 per-seq
        的，同批不同长度完全正确。SJF 最短序列优先（先完成释放 slot）。
        """
        selected = [s for s in self.running_sequences
                    if s.state == "decode" and not s.is_finished()]
        if not selected:
            return []

        # SJF 排序：短请求优先（先完成释放 slot），但全部进 batch
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
        """取变长 prefill batch（cu_seqlens 掩码，无需等长分组/padding）。

        seq 归属：chunked 进行中的 seq（state==prefill, prefill_done>0）留在 running，
        由本方法从 running 续切；新 prompt 从 waiting 取。

        变长一次性 prefill：从 waiting（或在途 chunk）按 FIFO 取尽可能多的 seq，受
        max_batch_size（并发数上限）和 max_prefill_tokens（单步 prefill 总 token 预算）约束。
        各 seq 长度可不同——prefill_runner 用 cu_seqlens 掩码处理，不再要求等长分组。
        长 prompt（剩余 > max_chunk_tokens）仍分块，单条切块期间留在 running 下步续切。
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

        # ---- 变长一次性 prefill：FIFO 取尽可能多 seq，受并发数 + 总 token 预算约束 ----
        # 注：不限制 running 总数（允许 running > max_batch，decode 仅取前 max_batch 条 SJF）。
        # 这样 prefill 快速清空 waiting 进入纯 decode 稳态，比零散穿插 prefill 吞吐更高
        #（显存由 KV 预算 n_blocks 兜底，block 不足时 alloc 失败由上层处理）。
        waiting.sort(key=lambda s: s.timestamp)  # FIFO
        selected = []
        total_tokens = 0
        for seq in waiting:
            if len(selected) >= self.max_batch_size:
                break
            remaining = seq.prefill_remaining
            # 短 prompt（≤ max_chunk_tokens）整条 prefill；长 prompt 走下方分块分支
            if remaining > self.max_chunk_tokens:
                continue
            if total_tokens + remaining > self.max_prefill_tokens:
                # 预算满：若已选了若干就停，否则（首条就超预算）仍放行首条避免饿死
                if selected:
                    break
            selected.append(seq)
            total_tokens += remaining

        if selected:
            for seq in selected:
                seq._chunk_len = seq.prefill_remaining
                seq._chunk_is_last = True
                self.waiting_queue.remove(seq)
                self.running_sequences.append(seq)
            logger.info(f"prefill selected: {len(selected)}, tokens: {total_tokens}, "
                        f"waiting_left: {len(self.waiting_queue)}")
            return selected

        # ---- 长 prompt 开始切块（batch=1）----
        long_seqs = [s for s in waiting if s.prefill_remaining > self.max_chunk_tokens]
        if long_seqs:
            seq = sorted(long_seqs, key=lambda s: s.timestamp)[0]
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
