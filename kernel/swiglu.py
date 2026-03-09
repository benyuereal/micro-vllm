import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    GateUp,      # 合并的 Gate/Up 张量指针
    Out,         # 输出张量指针
    n_elements,  # 输出的总元素数 (batch * seq * d)
    stride_d,    # 🔑 新增：gate_up 最后一维的大小 (即 2 * d)
    BLOCK_SIZE: tl.constexpr,
):
    """
    修正后的内存加载逻辑：
    适配 gate_up[:, :, :d] = up, gate_up[:, :, d:] = gate 的内存布局。
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # 输出的线性索引
    out_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n_elements

    # 修正策略：我们传进来的 stride_d 是 2d。
    # 我们在 Kernel 里计算 d = stride_d / 2 (因为是整数，我们用 >> 1)
    
    d = stride_d >> 1 # 假设 stride_d 是偶数 (这是必然的)
    
    row = out_offsets // d
    col = out_offsets % d
    
    gate_up_base = row * stride_d + col
    
    # 加载 Up (前半段) 和 Gate (后半段)
    up = tl.load(GateUp + gate_up_base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp + gate_up_base + d, mask=mask, other=0.0).to(tl.float32)

    # 计算 SiLU(gate) * up
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    out = silu_gate * up
    
    # 回写 (假设是 bfloat16，如果是 fp16 请修改)
    out = out.to(tl.bfloat16)
    tl.store(Out + out_offsets, out, mask=mask)


def swiglu_fused(
    gate_up: torch.Tensor,
) -> torch.Tensor:
    """
    接口保持不变。
    
    Args:
        gate_up: 形状为 (..., 2 * d) 的张量。
                 假设 gate_up[..., :d] 是 up，gate_up[..., d:] 是 gate。
    """
    assert gate_up.is_cuda, "Input must be on CUDA"
    
    original_shape = gate_up.shape
    last_dim = original_shape[-1]
    
    # 输出形状
    output_shape = original_shape[:-1] + (last_dim // 2,)
    
    # 展平
    gate_up_flat = gate_up.contiguous().view(-1)
    
    # 输出元素总数
    n_elements = gate_up_flat.numel() // 2
    
    # 创建输出
    out_flat = torch.empty(n_elements, dtype=gate_up.dtype, device=gate_up.device)

    # 配置
    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_up_flat,
        out_flat,
        n_elements,
        stride_d=last_dim,  # 🔑 传入最后一维的总长度 (2d)
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out_flat.view(output_shape)