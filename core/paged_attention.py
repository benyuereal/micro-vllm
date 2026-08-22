"""PagedAttention — 仅保留 RoPE cos/sin pool 预计算。

运行时 Qwen 通过 graph.attention._cos_pool / _sin_pool 取旋转表
（half-split 风格 [max_kv_capacity, rope_dim//2]）。decode/prefill 的
实际 attention 走 flash_attn_with_kvcache，不经过本类 forward。
"""
import torch
import torch.nn as nn


class PagedAttention(nn.Module):
    """分页注意力层。当前仅用作 RoPE cos/sin pool 的载体。"""

    def __init__(self, num_heads: int, head_size: int, kv_num_heads: int,
                 device: str = "auto", max_batch_size=16, max_blocks=32,
                 max_position=4096, max_tokens=8192, block_size=256,
                 rope_dim: int = None, rope_theta: float = 10000.0):
        super().__init__()
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.max_seq_blocks = (max_tokens + block_size - 1) // block_size
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        # RoPE 实际作用维度：GQA=head_size；MLA=qk_rope_head_dim（独立于 cache 存储维度）
        self.rope_dim = rope_dim if rope_dim is not None else head_size
        self.scale = head_size ** -0.5

        # 自动检测设备
        self.device = (torch.device('mps') if torch.backends.mps.is_available() else
                       torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        if device != "auto":
            self.device = torch.device(device)

        # 预计算 RoPE cos/sin pool（half-split：取前 dim//2 列）
        max_kv_capacity = max_blocks * 256
        dim = self.rope_dim
        base = float(rope_theta)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device).to(torch.bfloat16) / dim))
        t = torch.arange(max_kv_capacity, device=self.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_pool = emb.cos()[:max_kv_capacity, :dim // 2].contiguous()
        self._sin_pool = emb.sin()[:max_kv_capacity, :dim // 2].contiguous()
