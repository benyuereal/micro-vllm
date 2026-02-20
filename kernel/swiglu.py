"""
Fused SwiGLU activation kernel - 接收合并的 [up..., gate...]，计算 silu(gate) * up
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    GateUp,      # 合并的 [up..., gate...] 指针
    Out,         # 输出指针
    half_size,   # 一半的元素数量
    BLOCK_SIZE: tl.constexpr,
):
    """
    从合并的 GateUp 中加载 up 和 gate，计算 silu(gate) * up
    
    内存布局: [up_0, up_1, ..., up_n, gate_0, gate_1, ..., gate_n]
             |<--- half_size --->|<--- half_size --->|
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_size

    # up 在前半部分，gate 在后半部分
    up = tl.load(GateUp + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp + half_size + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # 输出: silu(gate) * up
    out = silu_gate * up

    # 写回
    tl.store(Out + offsets, out, mask=mask)


def swiglu_fused(gate_up: torch.Tensor) -> torch.Tensor:
    """
    接收合并的 gate_up: [..., 2*intermediate]，布局为 [up..., gate...]
    返回: [..., intermediate]
    """
    assert gate_up.is_cuda, "gate_up must be on CUDA device"
    
    *prefix, double_intermediate = gate_up.shape
    assert double_intermediate % 2 == 0, f"Last dim must be even, got {double_intermediate}"
    
    half = double_intermediate // 2

    # flatten
    gate_up_flat = gate_up.contiguous().view(-1)
    
    # 分配输出
    out = torch.empty(*prefix, half, dtype=gate_up.dtype, device=gate_up.device)

    # 总元素数 = batch * half
    n_elements = gate_up_flat.numel() // 2

    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_up_flat,
        out.view(-1),
        half,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out