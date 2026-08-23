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
                 device: str = "cuda", max_batch_size=16, max_blocks=32,
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

        self.device = torch.device(device)

        # 预计算 RoPE cos/sin pool（half-split：取前 dim//2 列）。
        # 关键：inv_freq / freqs 用 fp32 计算再转 bf16 存储。若全程 bf16 计算，base**(i/dim)
        # 在 bf16 下误差随位置累积（pos=400 时 cos 偏差达 0.33），长上下文下翻转 argmax
        # 导致输出退化。fp32 计算仅余 bf16 存储精度（~0.002），长上下文正确。
        max_kv_capacity = max_blocks * 256
        dim = self.rope_dim
        base = float(rope_theta)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device, dtype=torch.float32) / dim))
        t = torch.arange(max_kv_capacity, device=self.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_pool = emb.cos()[:max_kv_capacity, :dim // 2].to(torch.bfloat16).contiguous()
        self._sin_pool = emb.sin()[:max_kv_capacity, :dim // 2].to(torch.bfloat16).contiguous()
