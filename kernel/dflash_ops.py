"""DFlash2 投机解码公共算子。

- rope_half_split：RoPE（half-split / rotate_half，与 Qwen3 一致）。
  qwen3 adapter 的 prefill 路径按 per-token position gather 后逐 token 旋转。

纯 torch 实现（小 op，无现成 TileLang/Triton kernel，ROI 低）。
"""
import torch


def rope_half_split(x, cos, sin):
    """half-split RoPE（Llama 风格 rotate_half）：x [..., d]，cos/sin [..., d//2]。
    q/k 共用。返回旋转后的 x（同形状）。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
