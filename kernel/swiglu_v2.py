"""
SwiGLU 前向 (融合投影版 - 性能优化)
新增 fused_swiglu_fused：合并 Gate/Up 投影，减少一次 kernel launch
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
    """
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
    
    # SiLU: x * sigmoid(x)
    g_silu = g * tl.sigmoid(g)
    h = u * g_silu
    
    tl.store(hidden_ptr + h_offs, h.to(tl.float16), mask=mask)


def merge_gate_up_weights(gate_weight, up_weight):
    """
    合并 Gate 和 Up 的 weight 为 fused weight
    Args:
        gate_weight: [I, H] (如 [11008, 4096])
        up_weight: [I, H] (如 [11008, 4096])
    Returns:
        fused_weight: [2*I, H] (如 [22016, 4096])，已转置为 [H, 2*I] 供 matmul 使用
    """
    # 纵向拼接: [2*I, H]
    fused = torch.cat([gate_weight, up_weight], dim=0)
    # 转置为 [H, 2*I]，这样 matmul(x[H], fused_t[2*I,H]) 效率最高
    return fused.t().contiguous()


def fused_swiglu_fused(x, fused_gate_up_weight, down_weight):
    """
    SwiGLU 前向 (Gate/Up 融合投影版 - 性能优化)
    
    Args:
        x: [batch, seq_len, hidden_dim] 输入 (如 [1, 1, 4096])
        fused_gate_up_weight: [hidden_dim, 2*intermediate] 合并的 weight (如 [4096, 22016])
        down_weight: [hidden_dim, intermediate] down proj weight (如 [4096, 11008])
    Returns:
        output: [batch, seq_len, hidden_dim]
    """
    shape = x.shape
    x = x.view(-1, shape[-1])  # [M, H]
    M, H = x.shape
    
    # ✅ 关键优化: 一次 Matmul 完成 Gate + Up 投影
    # x[M, H] @ fused_weight[H, 2*I] -> gate_up[M, 2*I]
    # 比两次独立的 F.linear 节省一次 kernel launch 和内存带宽
    gate_up = torch.matmul(x, fused_gate_up_weight)
    
    # Split 成 gate 和 up (view 操作，无内存拷贝)
    gate, up = gate_up.chunk(2, dim=-1)
    
    # 确保 contiguous (Triton kernel 需要)
    gate = gate.contiguous()
    up = up.contiguous()
    
    M, I = gate.shape
    
    # Triton 融合 SiLU+Mul
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
    
    # Down projection (使用 F.linear 或 matmul)
    output = F.linear(hidden, down_weight)
    
    return output.view(shape)


# 保留原函数用于对比测试
def fused_swiglu(x, gate_weight, up_weight, down_weight):
    """
    SwiGLU 前向 (原始分离版)
    """
    shape = x.shape
    x = x.view(-1, shape[-1])
    
    # 1. 分离的 Gate 和 Up 投影 (2 次 kernel launch)
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    
    gate = gate.contiguous()
    up = up.contiguous()
    
    M, I = gate.shape
    
    # 2. Triton SiLU+Mul
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
    
    # 3. Down projection
    output = F.linear(hidden, down_weight)
    
    return output.view(shape)


# === 验证与性能测试 ===
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 模拟 7B 模型维度
    M, H, I = 1, 4096, 11008  # batch=1 (decode 场景), hidden=4096, intermediate=11008
    
    print("=" * 70)
    print("SwiGLU Gate/Up 融合优化测试")
    print("=" * 70)
    print(f"配置: batch={M}, hidden={H}, intermediate={I}")
    
    # 生成测试数据
    x = torch.randn((M, H), device='cuda', dtype=torch.float16)
    
    # 模拟 Qwen-7B 的 MLP weight
    gate_w = torch.randn((I, H), device='cuda', dtype=torch.float16)
    up_w = torch.randn((I, H), device='cuda', dtype=torch.float16)
    down_w = torch.randn((H, I), device='cuda', dtype=torch.float16)
    
    # 预处理：合并 weight（实际部署时只需做一次）
    fused_gate_up = merge_gate_up_weights(gate_w, up_w)
    print(f"\nWeight 合并: gate[{gate_w.shape}] + up[{up_w.shape}] -> fused[{fused_gate_up.shape}]")
    
    # Warmup
    for _ in range(10):
        _ = fused_swiglu(x, gate_w, up_w, down_w)
        _ = fused_swiglu_fused(x, fused_gate_up, down_w)
    torch.cuda.synchronize()
    
    # 正确性验证
    print("\n【1. 正确性验证】")
    y_old = fused_swiglu(x, gate_w, up_w, down_w)
    y_new = fused_swiglu_fused(x, fused_gate_up, down_w)
    
    max_err = torch.max(torch.abs(y_old - y_new)).item()
    mean_err = torch.mean(torch.abs(y_old - y_new)).item()
    print(f"  Max Error: {max_err:.6f}")
    print(f"  Mean Error: {mean_err:.6f}")
    print(f"  结果: {'✅ PASS' if max_err < 0.1 else '❌ FAIL'}")
    
    # 性能测试
    print("\n【2. 性能测试 (1000 iterations)】")
    iterations = 1000
    
    # 测试旧版 (分离 Gate/Up)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        y_old = fused_swiglu(x, gate_w, up_w, down_w)
    end.record()
    torch.cuda.synchronize()
    time_old = start.elapsed_time(end) / iterations
    
    # 测试新版 (融合 Gate/Up)
    start.record()
    for _ in range(iterations):
        y_new = fused_swiglu_fused(x, fused_gate_up, down_w)
    end.record()
    torch.cuda.synchronize()
    time_new = start.elapsed_time(end) / iterations
    
    print(f"\n  旧版 (分离投影): {time_old*1000:.3f} ms")
    print(f"  新版 (融合投影): {time_new*1000:.3f} ms")
    
    if time_new < time_old:
        speedup = time_old / time_new
        print(f"\n  ✅ 加速比: {speedup:.2f}x (快 {(speedup-1)*100:.1f}%)")
        print(f"     节省: {(time_old-time_new)*1000:.3f} ms/次")
    else:
        print(f"\n  ⚠️   slowdown: {time_new/time_old:.2f}x")
    
    print("\n" + "=" * 70)