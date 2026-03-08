from collections.abc import Mapping
from typing import Optional

import torch
import triton  # type: ignore[import]

import triton.language as tl  # type: ignore[import]

import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def scale_and_clamp(x, scale, dtype):
    """Scales a value and clamps it to the range of the target dtype.

    This function hard-wires the upper/lower bounds in order to be
    compatible with both `torch.compile` and `triton.jit`.
    """
    if dtype == tl.float8e4nv:
        clamp_min = -448.0
        clamp_max = 448.0
    elif dtype == tl.float8e5:
        clamp_min = -57344.0
        clamp_max = 57344.0
    elif dtype == tl.float16:
        clamp_min = -65504.0
        clamp_max = 65504.0
    elif dtype == tl.bfloat16:
        clamp_min = -3.3895313892515355e38
        clamp_max = 3.3895313892515355e38
    else:
        tl.static_assert(False, f"Unsupported dtype: {dtype}")

    return tl.clamp(x.to(tl.float32) * scale, clamp_min, clamp_max).to(dtype)

@triton.jit
def silu_and_mul_kernel(
    o_ptr,
    o_stride,
    o_scale_ptr,
    x_ptr,
    x_stride,
    x_scale_ptr,
    d,
    BLOCK_SIZE: tl.constexpr,
    HAS_X_SCALE: tl.constexpr,
    HAS_O_SCALE: tl.constexpr,
) -> None:
    """Sigmoid Linear Unit and Multiplication Kernel

    Args:
        o_ptr:       Pointer to the 2D output tensor.
        o_stride:    Output tensor stride.
        o_scale_ptr: The optional, known scale of the output activations.
        x_ptr:       Pointer to the 2D input tensor.
        x_stride:    Input tensor stride.
        x_scale_ptr: The optional, known scale of the input tensor.
        d:           The number of elements along the second dimension.
        BLOCK_SIZE:  Tunable block size to process in each kernel.

    Operating on a 2D grid, computes the following:

    ```
    out[i, j] = sigmoid(x[i, j]) * x[i, j] * x[i, j + d]
    ```

    If scales are provided, the input and output tensors are scaled.
    """

    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    if HAS_X_SCALE:
        x_scale = tl.load(x_scale_ptr)
        a *= x_scale
        b *= x_scale

    result = tl.sigmoid(a) * a * b

    if HAS_O_SCALE:
        o_scale = tl.load(o_scale_ptr)
        result = scale_and_clamp(result, o_scale, o_ptr.dtype.element_ty)

    tl.store(o_row_ptr + offsets, result, mask=mask)


def silu_and_mul(
        x: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        o_scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sigmoid Linear Unit and Multiplication
    支持 2D (Decode: [batch, 2*d]) 和 3D (Prefill: [batch, seq_len, 2*d]) 输入
    """
    # 🔑 关键改动：保存原始形状，处理 3D 输入
    original_shape = x.shape
    if x.ndim == 3:
        # Prefill 阶段：把 [batch, seq_len, 2*d] 展平为 [batch*seq_len, 2*d]
        x = x.reshape(-1, x.shape[-1])

    # 以下保持不变
    b, n = x.shape
    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    o = torch.empty((b, d), dtype=o_dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    silu_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        o_scale_ptr=o_scale,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=x_scale,
        d=d,
        BLOCK_SIZE=1024,
        HAS_X_SCALE=x_scale is not None,
        HAS_O_SCALE=o_scale is not None,
    )

    # 🔑 关键改动：恢复原始形状
    if len(original_shape) == 3:
        # Prefill 阶段：恢复为 [batch, seq_len, d]
        o = o.reshape(original_shape[0], original_shape[1], d)

    return o