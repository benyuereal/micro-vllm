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
import triton
import triton.language as tl

from kernel.gemv import gemv_or_matmul
from kernel.gemv_int8 import w8_linear


def _lin(x, w, out, env):
    """统一线性：w 为 bf16 [N,K] 或 (w_int8, scale) 元组（W8A16）。"""
    if isinstance(w, tuple):
        return w8_linear(x, w[0], w[1], out, env)
    return gemv_or_matmul(x, w, out, env)


@triton.jit
def _silu_mul_kernel(GU, OUT, M, INTER, stride_gu_m, stride_out_m,
                     BLOCK: tl.constexpr):
    """融合 silu(gate)*up：读 gate_up[:, :inter]=up, gate_up[:, inter:]=gate，
    写 out[:, :inter] = silu(gate)*up。一个 program 处理一行的一块。"""
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < INTER
    up = tl.load(GU + row * stride_gu_m + offs, mask=mask, other=0.0)
    gate = tl.load(GU + row * stride_gu_m + INTER + offs, mask=mask, other=0.0)
    # silu(x) = x * sigmoid(x)；bf16 下 tl.sigmoid 需 fp32，提升后乘 up 再回 bf16
    gate_f = gate.to(tl.float32)
    act = (gate_f * tl.sigmoid(gate_f) * up.to(tl.float32)).to(OUT.dtype.element_ty)
    tl.store(OUT + row * stride_out_m + offs, act, mask=mask)


def silu_mul_fused(gate_up: torch.Tensor, M: int, inter: int) -> torch.Tensor:
    """gate_up [M, 2*inter]（前半 up、后半 gate）→ act [M, inter] = silu(gate)*up。

    融合 silu+mul 单 kernel，替代 F.silu(gate)*up 的两个 elementwise kernel
    （profiled 5.9ms vs nano 融合 2.3ms/20 步，省 ~0.18ms/step）。"""
    out = torch.empty(M, inter, dtype=gate_up.dtype, device=gate_up.device)
    BLOCK = 1024
    grid = (M, triton.cdiv(inter, BLOCK))
    _silu_mul_kernel[grid](gate_up, out, M, inter,
                           gate_up.stride(0), out.stride(0), BLOCK=BLOCK)
    return out


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
    if w_is_nk:
        # [N,K] 布局：M=1 走 gemv_v2（W 直接读），否则 x @ W.t()。
        # x 可能是 3D（prefill [B,S,hidden]）→ 先展平 [M,hidden]，末尾 reshape 回原形。
        # W8A16：gu_w/d_w 可为 (w_int8, scale) 元组，N 从 w_int8.shape[0] 取。
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        M = m if m is not None else x2.shape[0]
        gu_n = gu_w[0].shape[0] if isinstance(gu_w, tuple) else gu_w.shape[0]
        d_n = d_w[0].shape[0] if isinstance(d_w, tuple) else d_w.shape[0]
        gate_up = torch.empty(M, gu_n, dtype=x.dtype, device=x.device)
        _lin(x2, gu_w, gate_up, "MICRO_GEMV_FFN")
        inter = gu_n // 2
        act = silu_mul_fused(gate_up, M, inter)
        out = torch.empty(M, d_n, dtype=x.dtype, device=x.device)
        out = _lin(act, d_w, out, "MICRO_GEMV_FFN")
        return out.reshape(*lead, d_n) if len(lead) > 1 else out
    else:
        # [K,N] 旧布局：x @ W（DeepSeek/老 Qwen 不变）
        gate_up = x @ gu_w
        up, gate = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ d_w
