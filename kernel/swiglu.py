# kernel/swiglu.py
"""
SwiGLU 前向 (分离优化版 - 修复数值误差)
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _silu_mul_kernel(
    gate_ptr, up_ptr, hidden_ptr,
    M, I,
    stride_gm, stride_gi,
    stride_um, stride_ui,
    stride_hm, stride_hi,
    BLOCK_SIZE: tl.constexpr = 2048,
):
    """
    纯 element-wise kernel: hidden = up * silu(gate)
    使用 2D 索引确保 stride 正确处理
    """
    pid = tl.program_id(0)
    
    # 每个线程处理 BLOCK_SIZE 个元素，展平索引
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 将 1D 展平索引转换为 2D (row, col) 以正确处理 stride
    rows = offs // I
    cols = offs % I
    
    mask = offs < (M * I)
    
    # 使用 stride 正确计算内存地址（支持非连续张量）
    g_offs = rows * stride_gm + cols * stride_gi
    u_offs = rows * stride_um + cols * stride_ui
    h_offs = rows * stride_hm + cols * stride_hi
    
    g = tl.load(gate_ptr + g_offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + u_offs, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU: x * sigmoid(x)
    g_silu = g * tl.sigmoid(g)
    h = u * g_silu
    
    tl.store(hidden_ptr + h_offs, h.to(tl.float16), mask=mask)


def fused_swiglu(x: torch.Tensor,
                 gate_weight: torch.Tensor,
                 up_weight: torch.Tensor,
                 down_weight: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU 前向 (分离架构优化版)
    """
    shape = x.shape
    x = x.view(-1, shape[-1])
    
    # 1. cuBLAS 计算 gate 和 up
    gate = F.linear(x, gate_weight)  # (M, I)
    up = F.linear(x, up_weight)      # (M, I)
    
    # ✅ 修复：确保张量是连续的（防止 stride 异常）
    gate = gate.contiguous()
    up = up.contiguous()
    
    M, I = gate.shape
    H = down_weight.shape[0]
    
    # 2. Triton 融合 SiLU+Mul
    hidden = torch.empty((M, I), device=x.device, dtype=x.dtype)
    
    total_elements = M * I
    grid = (triton.cdiv(total_elements, 2048),)
    
    _silu_mul_kernel[grid](
        gate, up, hidden,
        M, I,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE=2048,
        num_warps=4,
        num_stages=2,
    )
    
    # 3. cuBLAS 计算 down_proj
    output = F.linear(hidden, down_weight)  # (M, H)
    
    return output.view(shape)


def stable_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x.float()).half() if x.dtype == torch.float16 else F.silu(x)


# === 验证与性能测试 ===
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    M, H, I = 256, 4096, 11008
    
    print("=" * 70)
    print("SwiGLU 分离架构验证 (修复版)")
    print("=" * 70)
    
    def generate_realistic_data():
        x = torch.randn((M, H), device='cuda', dtype=torch.float32)
        x = x / (x.std() + 1e-6)
        
        weight_scale = 0.02
        gate_w = torch.randn((I, H), device='cuda', dtype=torch.float32) * weight_scale
        up_w = torch.randn((I, H), device='cuda', dtype=torch.float32) * weight_scale
        down_w = torch.randn((H, I), device='cuda', dtype=torch.float32) * weight_scale
        
        return x.half(), gate_w.half(), up_w.half(), down_w.half()
    
    x, gate_w, up_w, down_w = generate_realistic_data()
    
    print(f"\n配置: M={M}, H={H}, I={I}")
    print(f"输入 x: std={x.float().std():.3f}")
    
    # 1. 数值验证
    print("\n【1. 数值正确性验证】")
    y_new = fused_swiglu(x, gate_w, up_w, down_w)
    
    # PyTorch 参考实现（使用相同精度路径）
    with torch.no_grad():
        gate_ref = F.linear(x, gate_w)
        up_ref = F.linear(x, up_w)
        # 使用与 kernel 相同的计算路径：FP32 silu 然后转 FP16
        gate_fp32 = gate_ref.float()
        silu_gate = gate_fp32 * torch.sigmoid(gate_fp32)  # 显式 sigmoid 匹配 tl.sigmoid
        hidden_ref = up_ref.float() * silu_gate
        y_ref = F.linear(hidden_ref.half(), down_w)
    
    err = torch.max(torch.abs(y_new - y_ref)).item()
    mean_err = torch.mean(torch.abs(y_new - y_ref)).item()
    
    print(f"  Max Abs Error:  {err:.6f}")
    print(f"  Mean Abs Error: {mean_err:.6f}")
    print(f"  结果: {'✅ PASS' if err < 0.01 else '❌ FAIL'}")
    
    if err >= 0.01:
        print(f"\n  ⚠️  误差详情:")
        print(f"    误差位置: {torch.argmax(torch.abs(y_new - y_ref)).item()}")
        print(f"    y_new 范围: [{y_new.min():.3f}, {y_new.max():.3f}]")
        print(f"    y_ref 范围: [{y_ref.min():.3f}, {y_ref.max():.3f}]")
    
    # 2. 性能测试
    print("\n【2. 性能测试 (100 iterations)】")
    
    # 预热
    for _ in range(10):
        _ = fused_swiglu(x, gate_w, up_w, down_w)
        _ = F.linear(F.silu(F.linear(x, gate_w).float()).half() * F.linear(x, up_w), down_w)
    torch.cuda.synchronize()
    
    # 测试分离架构
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        y_fused = fused_swiglu(x, gate_w, up_w, down_w)
    end.record()
    torch.cuda.synchronize()
    time_new = start.elapsed_time(end) / 100
    
    # 测试 PyTorch 原生
    start.record()
    for _ in range(100):
        gate = F.linear(x, gate_w)
        up = F.linear(x, up_w)
        hidden = up * F.silu(gate.float()).half()
        y_torch = F.linear(hidden, down_w)
    end.record()
    torch.cuda.synchronize()
    time_torch = start.elapsed_time(end) / 100
    
    print(f"  PyTorch 原生:  {time_torch:.3f} ms")
    print(f"  分离架构:      {time_new:.3f} ms")
    
    if time_new < time_torch:
        speedup = time_torch / time_new
        print(f"  ✅ 加速比:      {speedup:.2f}x (快 {(speedup-1)*100:.1f}%)")
    else:
        print(f"  ⚠️   slowdown:   {time_new/time_torch:.2f}x")
    
    print("\n" + "=" * 70)
    if err < 0.01:
        print("✅ 分离架构验证通过，可安全用于生产环境")
        print(f"   加速比: {time_torch/time_new:.2f}x")
    else:
        print("❌ 数值误差过大，请检查实现")
    print("=" * 70)