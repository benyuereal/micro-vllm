"""
Fused SwiGLU activation kernel implemented in Triton.

SwiGLU (Swish-Gated Linear Unit) is used in LLaMA, PaLM, and other modern LLMs.
It's a variant of GLU that uses SiLU (Swish) as the activation function.

Full SwiGLU formula: y = silu(x @ W_gate) * (x @ W_up)

This kernel implements the fused activation part (after the linear projections):
y = silu(gate) * up

Where silu(x) = x * sigmoid(x)

Performance characteristics:
- Memory-bound operation (low arithmetic intensity ~1.3 FLOPs/byte)
- Fusing silu and multiply saves one memory round-trip
- Uses numerically stable sigmoid implementation

Reference: https://arxiv.org/abs/2204.02311 (PaLM paper)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    Gate,        # Gate tensor pointer (after W_gate projection)
    Up,          # Up tensor pointer (after W_up projection)
    Out,         # Output tensor pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused SwiGLU activation.

    Computes: out = silu(gate) * up = gate * sigmoid(gate) * up

    Memory access pattern:
    - Each program processes BLOCK_SIZE elements
    - Coalesced reads of gate and up tensors
    - Coalesced write of output
    - All operations are elementwise

    Arithmetic intensity:
    - Reads: N (gate) + N (up) = 2N elements
    - Writes: N (out) = N elements
    - FLOPs per element:
        - sigmoid: ~4 ops (exp, add, div) - but optimized
        - gate * sigmoid(gate): 1 mul
        - * up: 1 mul
    - Total: ~4 FLOPs per element
    - AI ≈ 4 / (3 * 2) = 0.67 for FP16 (memory-bound)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up values
    gate = tl.load(Gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # Compute output: silu(gate) * up
    out = silu_gate * up
    out = out.to(tl.bfloat16)
    # Store output
    tl.store(Out + offsets, out, mask=mask)



# ============================================================================
# 主入口函数
# ============================================================================

def swiglu_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU activation with adaptive configuration.

    Computes: y = silu(gate) * up

    This is the activation portion of SwiGLU, applied after the linear
    projections. The full SwiGLU in a transformer FFN is:
        y = silu(x @ W_gate) * (x @ W_up)

    This kernel fuses the silu and elementwise multiply, saving one
    memory round-trip compared to: F.silu(gate) * up

    Configuration is automatically selected based on tensor size:
    - Small tensors (< 32K): Smaller blocks, fewer warps
    - Medium tensors (32K-256K): Balanced config
    - Large tensors (> 256K): Larger blocks, more warps, pipelined

    Args:
        gate: Gate tensor of shape (...,), result of x @ W_gate projection.
        up: Up tensor of same shape as gate, result of x @ W_up projection.

    Returns:
        Output tensor of same shape as inputs.

    Example:
        >>> # In a transformer FFN:
        >>> gate = x @ W_gate  # Shape: (batch, seq, ffn_dim)
        >>> up = x @ W_up      # Shape: (batch, seq, ffn_dim)
        >>> y = swiglu_fused(gate, up)
    """
    assert gate.is_cuda, "Gate must be on CUDA device"
    assert up.is_cuda, "Up must be on CUDA device"
    assert gate.shape == up.shape, f"Shape mismatch: gate={gate.shape}, up={up.shape}"

    original_shape = gate.shape
    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    n_elements = gate_flat.numel()

    out_flat = torch.empty_like(gate_flat)

    # 🔑 根据元素数量选择最优配置
    BLOCK_SIZE, num_warps, num_stages = _get_swiglu_config(n_elements)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_flat,
        up_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out_flat.view(original_shape)


# ============================================================================
# 配置选择策略
# ============================================================================

def _get_swiglu_config(n_elements: int):
    """
    根据元素数量选择最优 kernel 配置

    优化原则：
    1. BLOCK_SIZE：影响每个 program 处理的元素数
       - 大 BLOCK：减少 program 数量，降低 launch 开销
       - 小 BLOCK：更多并行，但 overhead 增加

    2. num_warps：每个 program 实例的线程束数
       - 每个 warp 32 线程
       - 更多 warps = 更多并行，但寄存器压力增大
       - 内存受限的 kernel，warps 多不一定更快

    3. num_stages：内存流水线深度
       - 阶段化加载，隐藏内存延迟
       - 小 tensor 不需要多 stages

    返回: (BLOCK_SIZE, num_warps, num_stages)
    """
    # 元素数量阈值（基于典型 GPU 特性）

    # 极小 tensor (< 4K elements)
    # 单个 program 就够了，最小配置
    if n_elements < 4096:
        return (1024, 4, 1)

    # 小 tensor (4K - 32K elements)
    # 需要少量 programs，保守配置
    elif n_elements < 32 * 1024:
        return (1024, 4, 2)

    # 中等 tensor (32K - 256K elements)
    # 开始需要更多并行度
    elif n_elements < 256 * 1024:
        return (2048, 8, 2)

    # 大 tensor (256K - 1M elements)
    # 典型的 decode batch 场景
    elif n_elements < 1024 * 1024:
        return (4096, 8, 3)

    # 超大 tensor (> 1M elements)
    # Prefill 或大 batch 场景，最大化吞吐
    else:
        return (8192, 16, 4)

