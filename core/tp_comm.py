"""TP（张量并行）rank 间通信：bcast1（batch 元数据/seq 列表）+ bcast2（decode 采样 token）。

从 engine.py 抽出（行为不变）。TPCommunicator 不持有 engine 引用，依赖
（device / tokenizer / decode_ctx / scheduler）显式注入：
  - device: 常驻广播 buffer 所在设备
  - tokenizer: 完整 seq 列表广播/接收（BatchInferenceContext.broadcast_seqs/receive_seqs）
  - decode_ctx: bcast2 广播 rank0 本步采样 token（_decode_ctx.next_tokens）
  - scheduler: 常驻 token buffer 尺寸（max_batch_size）

紧凑协议（替代 broadcast_object_list 的 pickle 往返）：
  - bcast1: 单次 meta [batch_size, type_code, done, flag] GPU 张量广播（~0.1ms）。
    decode 稳态（batch 成员+顺序不变）flag=0，非 rank0 复用本地 seq store；
    batch 变化（seq 完成/新 prefill 转 decode）flag=1 + 完整 seq 列表；
    prefill 直接完整广播 seq 列表。
  - bcast2: decode 热路径只广播本步采样 token（[bs] GPU 张量），替代完整
    Sequence 广播（省 ~2.8ms/步 的 pickle+NCCL 往返）。prefill 批次回退完整 ctx 广播。
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from core.parallel_config import get_rank, get_world_size, rank0
from core.inference_context import BatchInferenceContext


class TPCommunicator:
    """TP rank 间通信器（bcast1/bcast2 紧凑协议）。world_size<=1 时所有方法 no-op。"""

    _TP_TYPE_CODE = {"decode": 0, "prefill": 1, "waiting": 2}
    _TP_TYPE_NAME = {0: "decode", 1: "prefill", 2: "waiting"}

    def __init__(self, device: str, tokenizer, decode_ctx, scheduler):
        self.device = device
        self.tokenizer = tokenizer
        self.decode_ctx = decode_ctx
        self.scheduler = scheduler
        # 常驻广播 buffer（lazy 分配，永不释放——dist.broadcast 是异步 GPU op，
        # 广播临时张量会在下一步张量被重新赋值/释放时 use-after-free）
        self._meta_buf_t: Optional[torch.Tensor] = None
        self._token_buf_t: Optional[torch.Tensor] = None
        # TP bcast1 紧凑协议：非 rank0 的本地 seq store（seq_id→Sequence）+ 上步
        # batch ids（判断 batch 是否变化）。rank0 仅用 _last_batch_ids。
        self._seq_store: Dict[int, "Sequence"] = {}
        self._last_batch_ids: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # 常驻 buffer
    # ------------------------------------------------------------------
    def _token_buf(self, bs: int) -> torch.Tensor:
        """常驻 [max_bs] int64 广播缓冲。dist.broadcast 是异步 GPU op，若直接广播
        sampler 返回的临时 next_tokens 张量，下一步该张量被重新赋值/释放而广播
        仍在读 → cudaErrorIllegalAddress（use-after-free）。广播常驻缓冲（永不释放，
        同 stream 上 copy→broadcast 有序）规避此问题。"""
        if self._token_buf_t is None:
            self._token_buf_t = torch.empty(
                self.scheduler.max_batch_size, dtype=torch.int64, device=self.device)
        return self._token_buf_t

    def _meta_buf(self) -> torch.Tensor:
        """常驻 [4] int64 广播缓冲：[batch_size, type_code, done, flag]。单次 GPU
        张量广播（~0.1ms，无 pickle），替代 broadcast_object_list 的 meta（~0.3ms +
        跨 rank 失步）。done 折进 meta（done 在 bcast1 前由 scheduler 状态算出）；
        flag 是 decode 的 batch-unchanged 标志（0=复用本地 seq store，1=完整广播
        seq 列表）。done+flag 都折进同一次广播，省掉每步两次独立小广播往返。"""
        if self._meta_buf_t is None:
            self._meta_buf_t = torch.empty(4, dtype=torch.int64, device=self.device)
        return self._meta_buf_t

    # ------------------------------------------------------------------
    # bcast2：decode 采样 token
    # ------------------------------------------------------------------
    def bcast_tokens(self, ctx: BatchInferenceContext):
        """TP bcast2：decode 热路径只广播本步采样 token（[bs] GPU 张量），
        替代完整 Sequence 广播（省 ~2.8ms/步 的 pickle+NCCL 往返）。
        prefill 批次无 decode next_tokens，回退完整 ctx 广播（非 rank0 需
        prefill_done/_next_token/state 推进）。"""
        if get_world_size() <= 1 or not rank0():
            return
        if ctx.batch_type == "decode":
            buf = self._token_buf(ctx.batch_size)
            buf[:ctx.batch_size].copy_(self.decode_ctx.next_tokens, non_blocking=True)
            dist.broadcast(buf[:ctx.batch_size], src=0)
        else:
            ctx.broadcast()

    def recv_tokens(self, ctx: BatchInferenceContext) -> List["Sequence"]:
        """TP bcast2 接收，返回供 update_sequences 使用的 seq 列表。
        decode：收 [bs] 采样 token 写回本地 seq._next_token（本地 seq 由 bcast1
        建立，update_sequences 本地 append 推进 output_ids/position/finished）；
        prefill：回退完整 receive（非 rank0 需 prefill_done/_next_token/state 推进）。"""
        if get_world_size() <= 1 or rank0():
            return ctx.sequences
        if ctx.batch_type == "decode":
            buf = self._token_buf(ctx.batch_size)
            dist.broadcast(buf[:ctx.batch_size], src=0)
            toks = buf[:ctx.batch_size].tolist()
            for i, seq in enumerate(ctx.sequences):
                seq._next_token = toks[i]
            return ctx.sequences
        recv = BatchInferenceContext.receive(self.tokenizer)
        return recv.sequences

    # ------------------------------------------------------------------
    # bcast1：batch 元数据 + seq 列表
    # ------------------------------------------------------------------
    def bcast_batch(self, ctx: BatchInferenceContext, done: bool = False):
        """TP bcast1（紧凑）：单次 meta [batch_size, type_code, done, flag] GPU
        张量广播（~0.1ms，无 pickle）。
        decode 稳态（batch 成员+顺序不变）flag=0，非 rank0 复用本地 seq store；
        batch 变化（seq 完成/新 prefill 转 decode）flag=1 + 完整 seq 列表；
        prefill 直接完整广播 seq 列表。
        替代每步 pickle 全部 32 个 Sequence（~2.0ms/步）。"""
        if get_world_size() <= 1 or not rank0():
            return
        meta = self._meta_buf()
        meta[0] = ctx.batch_size
        meta[1] = self._TP_TYPE_CODE[ctx.batch_type]
        meta[2] = 1 if done else 0
        if ctx.batch_type == "decode":
            ids = [s.seq_id for s in ctx.sequences]
            flag = 0 if ids == self._last_batch_ids else 1
            meta[3] = flag
            dist.broadcast(meta, src=0)
            if flag == 1:
                BatchInferenceContext.broadcast_seqs(ctx)
            self._last_batch_ids = ids
        else:
            meta[3] = 1
            dist.broadcast(meta, src=0)
            BatchInferenceContext.broadcast_seqs(ctx)

    def bcast_waiting(self, done: bool = False):
        """TP waiting：只广播 meta（type=waiting, done），非 rank0 据此空转。"""
        if get_world_size() <= 1 or not rank0():
            return
        meta = self._meta_buf()
        meta[0] = 0
        meta[1] = self._TP_TYPE_CODE["waiting"]
        meta[2] = 1 if done else 0
        meta[3] = 0
        dist.broadcast(meta, src=0)

    def recv_batch(self) -> Tuple[BatchInferenceContext, bool]:
        """TP bcast1 接收，返回 (ctx, done)。ctx 带 .sequences（供 step() 用），
        done 折在 meta[2]（省掉每步一次独立的 done 广播往返）。
        decode 稳态（flag=0）：复用本地 seq store（output_ids 已由 bcast2+
        update_sequences 同步，get_next_input_ids 即上步 token）；flag=1：
        完整 receive seq 列表并更新 store。prefill：完整 receive。"""
        if get_world_size() <= 1 or rank0():
            return None, False
        meta = self._meta_buf()
        dist.broadcast(meta, src=0)
        bs = int(meta[0].item())
        batch_type = self._TP_TYPE_NAME[int(meta[1].item())]
        done = bool(int(meta[2].item()))
        ctx = BatchInferenceContext(bs, batch_type)
        if batch_type == "waiting":
            return ctx, done
        if batch_type == "decode":
            if int(meta[3].item()) == 0:
                # 稳态：复用本地 store（ids 与 rank0 相同，含 padding 重复）
                ctx.sequences = [self._seq_store[sid] for sid in self._last_batch_ids]
            else:
                seqs = BatchInferenceContext.receive_seqs(bs, self.tokenizer)
                for s in seqs:
                    self._seq_store[s.seq_id] = s
                self._last_batch_ids = [s.seq_id for s in seqs]
                ctx.sequences = seqs
        else:
            seqs = BatchInferenceContext.receive_seqs(bs, self.tokenizer)
            for s in seqs:
                self._seq_store[s.seq_id] = s
            ctx.sequences = seqs
        return ctx, done
