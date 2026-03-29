import torch
from typing import List, Optional


class DecodeContext:
    """
    持久化 decode 阶段的跨步批次状态。

    - prepare()：步开始时调用，返回 input_ids（batch 不变时返回 None）
    - commit()：采样后调用，GPU→GPU 预填充 + D2H + 更新 seq._next_token
    """

    __slots__ = ('seq_ids', 'temps', 'topp', 'next_tokens')

    def __init__(self):
        self.seq_ids:     Optional[List[int]]   = None
        self.temps:       Optional[torch.Tensor] = None  # [bs] float, on device
        self.topp:        Optional[torch.Tensor] = None  # [bs] float, on device
        self.next_tokens: Optional[torch.Tensor] = None  # [bs] int,   on device

    def prepare(self, batch, device: str) -> Optional[torch.Tensor]:
        """
        刷新批次状态，返回本步的 input_ids。

        - batch 不变 → 复用 temps/topp，返回 None（上一步 commit 已预填充 _input_ids）
        - batch 变化 → 重建 temps/topp，从 CPU 构建 input_ids tensor
        """
        cur_ids = [seq.seq_id for seq in batch]
        if cur_ids == self.seq_ids:
            return None

        self.seq_ids = cur_ids
        self.temps = torch.tensor([seq.temperature for seq in batch], device=device)
        self.topp  = torch.tensor([seq.top_p for seq in batch], device=device)
        return torch.tensor(
            [seq.get_next_input_ids() for seq in batch], device=device
        ).squeeze(1)

    def commit(self, next_tokens_gpu: torch.Tensor,
               input_ids_buf: torch.Tensor,
               batch_size: int, batch) -> List[int]:
        """
        采样后维护上下文状态：
          1. GPU→GPU 预填充下一步 input_ids（非阻塞）
          2. D2H，将 token 写回每个 seq._next_token
        """
        input_ids_buf[:batch_size].copy_(next_tokens_gpu, non_blocking=True)
        self.next_tokens = next_tokens_gpu

        next_tokens = next_tokens_gpu.tolist()
        for i, seq in enumerate(batch):
            seq._next_token = next_tokens[i]
        return next_tokens
