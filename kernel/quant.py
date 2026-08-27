"""W8A16 量化：权重 INT8（per-channel / per-output-row）+ 激活 bf16。

W8A16 的收益在 decode（memory-bound）：权重字节数减半 → GEMV 带宽减半。
per-channel scale：scale[n] = max(|w[n,:]|) / 127，w_int8[n,:] = round(w[n,:] / scale[n])。
反量化：w_bf16[n,:] = w_int8[n,:] * scale[n]。

GEMM：out[n] = scale[n] * sum_k x[k] * w_int8[n,k]（fp32 累加，scale 最后乘）。
"""
import torch


def quantize_per_channel(w: torch.Tensor):
    """w [N,K] bf16/fp32 → (w_int8 [N,K] int8, scale [N] fp32)。per-output-row。"""
    w = w.float()
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # [N,1]
    scale = amax / 127.0                                       # [N,1]
    w_int8 = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return w_int8, scale.squeeze(1).contiguous()               # scale [N]


def dequantize_per_channel(w_int8: torch.Tensor, scale: torch.Tensor,
                           dtype=torch.bfloat16) -> torch.Tensor:
    """w_int8 [N,K] + scale [N] → w [N,K] dtype。"""
    return (w_int8.float() * scale.unsqueeze(1)).to(dtype)
