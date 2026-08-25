import torch
from typing import List, Optional


class DecodeContext:
    """
    持久化 decode 阶段的跨步批次状态。

    - prepare()：步开始时调用，返回 input_ids（batch 不变时返回 None）
    - commit()：采样后调用，GPU→GPU 预填充 + D2H + 更新 seq._next_token
    """

    __slots__ = ('seq_ids', 'temps', 'topp', 'rep_penalties', 'prev_tokens', 'next_tokens',
                 'all_greedy', 'any_rep_pen')

    def __init__(self):
        self.seq_ids:     Optional[List[int]]   = None
        self.temps:       Optional[torch.Tensor] = None  # [bs] float, on device
        self.topp:        Optional[torch.Tensor] = None  # [bs] float, on device
        self.rep_penalties: Optional[torch.Tensor] = None  # [bs] float, on device
        self.prev_tokens: Optional[torch.Tensor] = None  # [bs, L] int, 历史token（-1 padding）
        self.next_tokens: Optional[torch.Tensor] = None  # [bs] int,   on device
        # CPU 侧预判标志（避免 sampler 里 torch.any 的 GPU→CPU 同步）
        self.all_greedy: bool = False
        self.any_rep_pen: bool = False

    def prepare(self, batch, device: str, cache_manager,
                batch_dirty: bool = True) -> Optional[torch.Tensor]:
        """
        刷新批次状态，返回本步的 input_ids；同时驱动 cache_manager 更新缓存元数据。

        - batch 不变 → 复用 temps/topp/rep_penalties，返回 None
        - batch 变化 → 重建 temps/topp/rep_penalties，返回新 input_ids
        两种情况都通过 cache_manager.prepare(batch_switched) 统一处理缓存更新。

        batch_dirty 由 engine 维护的脏标志（避免每步重建 512 元素 cur_ids/ctx_lens
        列表 + 列表比较的 ~1.2ms CPU 开销）：稳定 decode 下每步 batch 成员与顺序
        完全不变，仅当有序列完成 / prefill 新进 / append 跨 block 分配时才置脏。
        """
        if not batch_dirty and not cache_manager._dirty_seqs:
            # 稳态：batch 不变且无新 block 分配 → block_table 不变、seqlens 由
            # commit GPU 原地维护，无需任何列表构建或 cache_manager 更新。
            return None

        cur_ids = [seq.seq_id for seq in batch]
        batch_switched = (cur_ids != self.seq_ids)
        ctx_lens = [seq.current_position for seq in batch]

        cache_manager.prepare(
            cur_ids,
            ctx_lens,
            batch_switched,
        )

        if not batch_switched:
            return None

        self.seq_ids = cur_ids
        temps_list = [seq.temperature for seq in batch]
        rep_list = [getattr(seq, 'repetition_penalty', 1.0) for seq in batch]
        self.temps = torch.tensor(temps_list, device=device)
        self.topp  = torch.tensor([seq.top_p for seq in batch], device=device)
        self.rep_penalties = torch.tensor(rep_list, device=device)
        # CPU 侧预判（避免 sampler torch.any 同步）：全 greedy / 有 rep penalty
        self.all_greedy = all(t <= 0 for t in temps_list)
        self.any_rep_pen = any(r > 1.0 for r in rep_list)
        # 历史 token（prompt + 已生成），仅当任一 seq 启用 repetition penalty 时才构造
        # （[bs, L] tensor 构造 + H2D 开销可观，greedy/无惩罚场景跳过）。-1 padding 到等长。
        # 用 CPU 侧 any_rep_pen 避免 rep_penalties.max() 的 GPU→CPU 同步。
        if self.any_rep_pen:
            hist = [list(seq.input_ids) + list(seq.output_ids) for seq in batch]
            max_l = max(len(h) for h in hist)
            padded = [h + [-1] * (max_l - len(h)) for h in hist]
            self.prev_tokens = torch.tensor(padded, dtype=torch.long, device=device)
        else:
            self.prev_tokens = None
        return torch.tensor(
            [seq.get_next_input_ids() for seq in batch], device=device
        ).squeeze(1)

    def commit(self, next_tokens_gpu: torch.Tensor,
               input_ids_buf: torch.Tensor,
               batch_size: int, batch) -> List[int]:
        """
        采样后维护上下文状态（rank0 only）：
          1. GPU→GPU 预填充下一步 input_ids（非阻塞）
          2. D2H，将 token 写回每个 seq._next_token
        注：cache_seqlens +1 由 _decode 直接调用 cache_manager.commit，在所有 rank 上执行。
        """
        input_ids_buf[:batch_size].copy_(next_tokens_gpu, non_blocking=True)
        self.next_tokens = next_tokens_gpu

        next_tokens = next_tokens_gpu.tolist()
        # decode batch 含循环复制的 pad 重复 seq（同一 Sequence 对象出现多次）。
        # 仅对每个 seq_id 的首次出现写 _next_token：pad 行在 GDN 架构下被 kernel
        # 跳过（IS_REAL=0），其输出是未初始化垃圾，若按行写会覆盖真实 token。
        # 非 GDN 架构 pad 行输出与真实行相同，去重无副作用。
        seen = set()
        for i, seq in enumerate(batch):
            if seq.seq_id in seen:
                continue
            seen.add(seq.seq_id)
            seq._next_token = next_tokens[i]
        return next_tokens
