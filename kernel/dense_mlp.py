"""Dense SwiGLU MLP 公共算子。

DeepSeek / Qwen 的 dense MLP 都是 SwiGLU: out = (silu(gate) * up) @ d_w。

权重布局（两套，由 w_is_nk 参数区分）：
  - w_is_nk=True（Qwen3 新布局，GEMV 友好，零额外显存）：
      gu_w = cat([up, gate], dim=0) → [2*inter, hidden]=[N,K]，每输出行连续 K
      d_w  = [hidden, inter]=[N,K]
      计算：x @ W.t()（M>1）/ gemv_v2(x, W)（M=1，手写 kernel 直接读 [N,K]，快 cuBLAS 32-44%）
  - w_is_nk=False（DeepSeek / 老 Qwen 旧布局，保持不变）：
      gu_w = cat([up, gate]).t() → [hidden, 2*inter]=[K,N]
      d_w  = [inter, hidden]=[K,N]（DeepSeek）/ [hidden, inter]=[K,N]（老 Qwen）
      计算：x @ W（旧逻辑，不改）

DeepSeek MoE expert 权重 _e_gu 另用 [gate|up]，TileLang moe kernel 内部写死，与本文件无关。
"""
import os
import torch
import torch.nn.functional as F

from kernel.gemv import gemv_v2, gemv_available


def dense_swiglu(x, gu_w, d_w, m=None, w_is_nk=False):
    """Dense SwiGLU: out = (silu(gate) * up) @ d_w。

    Args:
        x:       [M, hidden]        normed 输入
        gu_w:    gate_up 权重（布局见 w_is_nk）
        d_w:     down 权重（布局见 w_is_nk）
        m:       int 或 None        M=x.shape[0]；显式传入避免 prefill reshape 后误判
        w_is_nk: bool               True=权重 [N,K]（Qwen3，GEMV 友好）；
                                   False=权重 [K,N]（DeepSeek/老 Qwen 旧布局）
    Returns:
        out:     [M, hidden]
    """
    if m is None:
        m = x.shape[0]
    use_gemv = (m == 1 and gemv_available() and os.environ.get("MICRO_GEMV_FFN", "1") != "0")
    if w_is_nk:
        # [N,K] 布局：M=1 走 gemv_v2（W 直接读），否则 x @ W.t()
        gate_up = gemv_v2(x, gu_w) if use_gemv else x @ gu_w.t()
        up, gate = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        return gemv_v2(act, d_w) if use_gemv else act @ d_w.t()
    else:
        # [K,N] 旧布局：x @ W（DeepSeek/老 Qwen 不变）
        gate_up = x @ gu_w
        up, gate = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ d_w
