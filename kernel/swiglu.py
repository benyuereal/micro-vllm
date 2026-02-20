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
    # sigmoid(x) = 1 / (1 + exp(-x))
    # For numerical stability, use: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    # But the standard form works fine with Triton's exp implementation
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate

    # Compute output: silu(gate) * up
    out = silu_gate * up

    # Store output
    tl.store(Out + offsets, out, mask=mask)


def swiglu_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU activation.

    Computes: y = silu(gate) * up

    This is the activation portion of SwiGLU, applied after the linear
    projections. The full SwiGLU in a transformer FFN is:
        y = silu(x @ W_gate) * (x @ W_up)

    This kernel fuses the silu and elementwise multiply, saving one
    memory round-trip compared to: F.silu(gate) * up

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

    # Flatten for kernel
    original_shape = gate.shape
    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    n_elements = gate_flat.numel()

    # Allocate output
    out_flat = torch.empty_like(gate_flat)

    # Launch kernel
    # FIXME: smaller block size might be better for small tensors
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_flat,
        up_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.view(original_shape)

