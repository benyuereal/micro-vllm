"""
SwiGLU 前向 (融合投影修复版)
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
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    rows = offs // I
    cols = offs % I
    mask = offs < (M * I)
    
    g_offs = rows * stride_gm + cols * stride_gi
    u_offs = rows * stride_um + cols * stride_ui
    h_offs = rows * stride_hm + cols * stride_hi
    
    g = tl.load(gate_ptr + g_offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + u_offs, mask=mask, other=0.0).to(tl.float32)
    
    g_silu = g * tl.sigmoid(g)
    h = u * g_silu
    
    tl.store(hidden_ptr + h_offs, h.to(tl.float16), mask=mask)


def merge_gate_up_weights(gate_weight, up_weight):
    """
    合并 Gate 和 Up 的 weight
    修复：确保数值范围和内存布局
    """
    # 确保是 fp16 且 contiguous
    gate_f16 = gate_weight.contiguous()
    up_f16 = up_weight.contiguous()
    
    # 纵向拼接: [2*I, H]
    fused = torch.cat([gate_f16, up_f16], dim=0)
    
    # 转置为 [H, 2*I] 并确保 contiguous
    return fused.t().contiguous()


def fused_swiglu_fused(x, fused_gate_up_weight, down_weight):
    shape = x.shape
    x = x.view(-1, shape[-1])
    M, H = x.shape
    
    # 确保输入是 contiguous 的 fp16
    x = x.contiguous()
    
    # 一次 Matmul: [M, H] @ [H, 2*I] -> [M, 2*I]
    gate_up = torch.matmul(x, fused_gate_up_weight)
    
    # Chunk: 注意 chunk 返回的是 view，需要显式 contiguous
    gate, up = gate_up.chunk(2, dim=-1)
    
    # ✅ 修复：必须 contiguous 才能传给 Triton
    gate = gate.contiguous()
    up = up.contiguous()
    
    M, I = gate.shape
    
    # Triton kernel
    hidden = torch.empty((M, I), device=x.device, dtype=torch.float16)
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
    
    # Down proj
    output = F.linear(hidden, down_weight)
    return output.view(shape)


def fused_swiglu(x, gate_weight, up_weight, down_weight):
    """原始分离版"""
    shape = x.shape
    x = x.view(-1, shape[-1])
    
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    
    gate = gate.contiguous()
    up = up.contiguous()
    
    M, I = gate.shape
    
    hidden = torch.empty((M, I), device=x.device, dtype=torch.float16)
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
    
    output = F.linear(hidden, down_weight)
    return output.view(shape)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    M, H, I = 1, 4096, 11008
    
    print("=" * 70)
    print("SwiGLU 修复测试")
    print("=" * 70)
    
    # ✅ 修复：使用小数值初始化，避免 FP16 溢出
    x = torch.randn((M, H), device='cuda', dtype=torch.float16) * 0.01
    
    gate_w = torch.randn((I, H), device='cuda', dtype=torch.float16) * 0.01
    up_w = torch.randn((I, H), device='cuda', dtype=torch.float16) * 0.01
    down_w = torch.randn((H, I), device='cuda', dtype=torch.float16) * 0.01
    
    print(f"输入 x 范围: [{x.min():.3f}, {x.max():.3f}]")
    
    # 合并 weight
    fused_gate_up = merge_gate_up_weights(gate_w, up_w)
    print(f"Fused weight 形状: {fused_gate_up.shape}")
    print(f"Fused weight 范围: [{fused_gate_up.min():.3f}, {fused_gate_up.max():.3f}]")
    
    # ✅ 修复：充分 Warmup（包括 Triton 编译）
    print("\nWarming up...")
    for i in range(20):
        _ = fused_swiglu(x, gate_w, up_w, down_w)
        _ = fused_swiglu_fused(x, fused_gate_up, down_w)
        if i == 10:
            torch.cuda.synchronize()  # 确保编译完成
    
    torch.cuda.synchronize()
    print("Warmup done")
    
    # 正确性验证
    print("\n【1. 正确性验证】")
    with torch.no_grad():
        y_old = fused_swiglu(x, gate_w, up_w, down_w)
        y_new = fused_swiglu_fused(x, fused_gate_up, down_w)
    
    # 检查 NaN
    if torch.isnan(y_old).any():
        print("  ❌ 旧版出现 NaN")
    if torch.isnan(y_new).any():
        print("  ❌ 新版出现 NaN")
        # 打印中间值调试
        gate_up = torch.matmul(x, fused_gate_up)
        print(f"    gate_up range: [{gate_up.min():.3f}, {gate_up.max():.3f}]")
        gate, up = gate_up.chunk(2, dim=-1)
        print(f"    gate range: [{gate.min():.3f}, {gate.max():.3f}]")
        print(f"    up range: [{up.min():.3f}, {up.max():.3f}]")
    else:
        max_err = torch.max(torch.abs(y_old - y_new)).item()
        print(f"  ✅ Max Error: {max_err:.6f}")
    
    # 性能测试（减小迭代次数，避免时间过长）
    print("\n【2. 性能测试 (100 iterations)】")
    iterations = 100
    
    # 清理缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 测试旧版
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        y_old = fused_swiglu(x, gate_w, up_w, down_w)
    end.record()
    torch.cuda.synchronize()
    time_old = start.elapsed_time(end) / iterations
    
    # 测试新版
    start.record()
    for _ in range(iterations):
        y_new = fused_swiglu_fused(x, fused_gate_up, down_w)
    end.record()
    torch.cuda.synchronize()
    time_new = start.elapsed_time(end) / iterations
    
    print(f"\n  旧版 (分离): {time_old*1000:.3f} ms")
    print(f"  新版 (融合): {time_new*1000:.3f} ms")
    
    if time_new < time_old:
        speedup = time_old / time_new
        print(f"\n  ✅ 加速比: {speedup:.2f}x")
    else:
        print(f"\n  ⚠️  slowdown: {time_new/time_old:.2f}x")
    
    print("\n" + "=" * 70)