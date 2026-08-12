"""Dense SwiGLU MLP 公共算子。

DeepSeek / Qwen 的 dense MLP 都是 SwiGLU: out = (silu(gate) * up) @ d_w，
权重预合并 gu_w = cat([up, gate]).t()  形状 [hidden, 2*inter]（up 在前半、gate 在后半）。

两架构 dense 层统一此布局：
  - Qwen 原生 w1=up, w2=gate → cat([w1, w2]).t() 天然 [up|gate]。
  - DeepSeek 原生 gate_proj/up_proj → prepare_weights 显式 cat([up, gate]).t() 对齐。
（DeepSeek MoE expert 权重 _e_gu 另用 [gate|up]，TileLang moe kernel 内部写死，
 与本文件无关——dense 与 MoE 权重独立。）
"""
import torch
import torch.nn.functional as F


def dense_swiglu(x, gu_w, d_w):
    """Dense SwiGLU: out = (silu(gate) * up) @ d_w。

    Args:
        x:    [N, hidden]           normed 输入
        gu_w: [hidden, 2*inter]     cat([up, gate]).t()，前半 up、后半 gate
        d_w:  [hidden, inter]       down 投影（已 .t()）
    Returns:
        out:  [N, hidden]
    """
    gate_up = x @ gu_w
    up, gate = gate_up.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ d_w
