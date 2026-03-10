import torch
import triton
import triton.language as tl
import time

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



# ============================================================================
# 【新增代码：优化后的 SwiGLU 版本 (预分配缓冲区)】
# ============================================================================
def swiglu_gemm(
        gate_up: torch.Tensor,
        out_buffer: torch.Tensor,  # 🔑 新增：预分配的输出缓冲区
) -> torch.Tensor:
    """
    【新增】优化后的 SwiGLU，专为贴边融合设计

    Args:
        gate_up: 形状为 (..., 2 * d) 的张量
        out_buffer: 预分配的输出缓冲区 (下一个 Matmul 的输入)

    Returns:
        out_buffer: 直接返回预分配的缓冲区
    """
    assert gate_up.is_cuda and out_buffer.is_cuda, "All must be on CUDA"
    assert gate_up.dtype == out_buffer.dtype, f"Dtype mismatch: {gate_up.dtype} vs {out_buffer.dtype}"

    original_shape = gate_up.shape
    last_dim = original_shape[-1]

    # 验证输出缓冲区形状
    expected_output_shape = original_shape[:-1] + (last_dim // 2,)
    assert out_buffer.shape == expected_output_shape, f"Shape mismatch: {out_buffer.shape} vs {expected_output_shape}"

    # 展平
    gate_up_flat = gate_up.contiguous().view(-1)
    out_flat = out_buffer.contiguous().view(-1)

    # 输出元素总数
    n_elements = out_flat.numel()

    # 配置 (和原有代码保持一致)
    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_up_flat,
        out_flat,
        n_elements,
        stride_d=last_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out_buffer


def benchmark_swiglu():
    """
    性能对比测试：对比原有 swiglu_fused 和新增的优化版本
    模拟真实的 Decode 场景 MLP 完整路径：
    GateUp Matmul -> SwiGLU -> Down Matmul
    """
    # Decode 阶段典型配置
    batch_size = 64
    seq_len = 1
    hidden_size = 4096
    device = "cuda"
    dtype = torch.bfloat16

    # 初始化数据
    normed = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    mlp_gu = torch.randn(hidden_size, 2 * hidden_size, device=device, dtype=dtype)
    mlp_d = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)

    # 预分配缓冲区 (用于优化版本)
    gate_up_buffer = torch.empty(batch_size, 2 * hidden_size, device=device, dtype=dtype)
    swiglu_out_buffer = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
    mlp_out_buffer = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # 方案 1: 原有 swiglu_fused
    # -------------------------------------------------------------------------
    def original_path():
        # 1. GateUp Matmul
        gate_up = normed @ mlp_gu
        # 2. 原有 SwiGLU (动态分配内存)
        activated = swiglu_fused(gate_up)
        # 3. Down Matmul
        mlp_out = activated @ mlp_d
        return mlp_out

    # -------------------------------------------------------------------------
    # 方案 2: 新增的 swiglu_fused_for_gemm
    # -------------------------------------------------------------------------
    def fused_path():
        # 1. GateUp Matmul (预分配缓冲区)
        torch.matmul(normed, mlp_gu, out=gate_up_buffer)
        # 2. 优化后的 SwiGLU (预分配缓冲区)
        activated = swiglu_gemm(gate_up_buffer, swiglu_out_buffer)
        # 3. Down Matmul (预分配缓冲区)
        torch.matmul(activated, mlp_d, out=mlp_out_buffer)
        return mlp_out_buffer

    # ============================================================================
    # 预热
    # ============================================================================
    print("Warming up...")
    for _ in range(30):
        original_path()
        fused_path()
    torch.cuda.synchronize()

    # ============================================================================
    # 性能测试
    # ============================================================================
    num_iters = 2000
    print(f"Running benchmark ({num_iters} iterations)...")

    # 测试原有方法
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        original_path()
    torch.cuda.synchronize()
    t1 = time.time()
    original_time = (t1 - t0) / num_iters * 1000  # ms

    # 测试新方法
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iters):
        fused_path()
    torch.cuda.synchronize()
    t1 = time.time()
    fused_time = (t1 - t0) / num_iters * 1000  # ms

    # ============================================================================
    # 打印结果
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"🔥 SwiGLU 性能对比 (Decode场景: Batch={batch_size}, Hidden={hidden_size})")
    print("=" * 80)
    print(f"{'方案':<25} | {'耗时 (ms)':<12} | {'相对加速':<10}")
    print("-" * 80)
    print(f"{'原有 swiglu_fused':<25} | {original_time:<12.4f} | {'1.00x':<10}")
    print(f"{'新增 swiglu_fused_for_gemm':<25} | {fused_time:<12.4f} | {original_time / fused_time:<10.2f}x")
    print("=" * 80)
    print(f"💡 优化效果：")
    print(f"   - 耗时降低: {(1 - fused_time / original_time) * 100:.1f}%")
    print("=" * 80)

    # 验证正确性
    mlp_original = original_path()
    mlp_fused = fused_path()
    is_correct = torch.allclose(mlp_original, mlp_fused, rtol=1e-3, atol=1e-3)
    print(f"\n✅ 正确性验证: {is_correct}")
    if not is_correct:
        print(f"   最大误差: {torch.max(torch.abs(mlp_original - mlp_fused))}")


if __name__ == "__main__":
    benchmark_swiglu()