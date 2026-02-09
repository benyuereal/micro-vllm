"""
Fused Add + RMSNorm
融合残差连接和 RMSNorm，减少一次内存读写
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def add_rms_norm_fwd(
    Residual, X, Y, W,
    M, N,
    stride_r, stride_x, stride_y,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Kernel: Y = RMSNorm(Residual + X) * W
    同时 Residual 会被更新为 (Residual + X) 用于后续残差连接
    """
    row = tl.program_id(0)
    if row >= M:
        return
    
    # 行指针
    R_row = Residual + row * stride_r
    X_row = X + row * stride_x
    Y_row = Y + row * stride_y
    
    # 第一次遍历：计算 (residual + x) 的平方和
    rms_sq = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        
        # 融合加法
        sum_val = r + x
        rms_sq += tl.sum(sum_val * sum_val)
    
    # 计算 rstd
    rstd = 1.0 / tl.sqrt(rms_sq / N + eps)
    
    # 第二次遍历：归一化 + 乘 weight + 写回
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        
        # 再次计算 sum (避免存储中间结果，省内存)
        sum_val = r + x
        
        # RMSNorm + weight
        y = sum_val * rstd * w
        
        # 写回输出
        tl.store(Y_row + cols, y, mask=mask)
        
        # 可选：将 sum 写回 Residual (作为 mlp_res 保存)
        # 如果不需要在 Residual 中保存 sum，可以注释掉这行
        tl.store(R_row + cols, sum_val.to(r.dtype), mask=mask)


def fused_add_rms_norm(residual: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    融合 Add + RMSNorm
    
    Args:
        residual: [M, N] 残差输入 (会被修改为 residual + x)
        x: [M, N] 待加输入 (如 attn_proj)
        weight: [N] RMSNorm weight
        eps: 防止除零
    
    Returns:
        y: [M, N] RMSNorm 后的结果
        同时 residual 会被更新为 (residual + x)，可直接作为 mlp_res 使用
    """
    # 确保 contiguous
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    
    M, N = residual.shape
    
    # 输出 buffer
    y = torch.empty_like(residual)
    
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)
    
    add_rms_norm_fwd[(M,)](
        residual, x, y, weight,
        M, N,
        residual.stride(0), x.stride(0), y.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 512 else 8,
    )
    
    return y


# === 验证 ===
if __name__ == "__main__":
    torch.manual_seed(0)
    
    M, N = 4, 4096
    residual = torch.randn((M, N), device='cuda', dtype=torch.float16) * 0.01
    x = torch.randn((M, N), device='cuda', dtype=torch.float16) * 0.01
    weight = torch.ones((N,), device='cuda', dtype=torch.float16)
    
    # 保存原始 residual 用于对比
    residual_ref = residual.clone()
    
    # Fused 版本
    y_fused = fused_add_rms_norm(residual, x, weight, eps=1e-6)
    
    # 分离版本 (参考实现)
    sum_ref = residual_ref + x
    variance = sum_ref.float().pow(2).mean(-1, keepdim=True)
    y_ref = sum_ref * torch.rsqrt(variance + 1e-6) * weight
    
    # 验证数值
    err = torch.max(torch.abs(y_fused.float() - y_ref.float())).item()
    print(f"Max Error: {err:.2e}")
    print(f"Result: {'✅ PASS' if err < 1e-3 else '❌ FAIL'}")

    # 验证 residual 是否被正确更新
    residual_expected = residual_ref + x
    residual_err = torch.max(torch.abs(residual.float() - residual_expected.float())).item()
    print(f"Residual Update Error: {residual_err:.2e}")
    print(f"Residual Update: {'✅ PASS' if residual_err < 1e-3 else '❌ FAIL'}")

    # 性能测试
    print("\n=== 性能测试 ===")
    import time
    
    # Warmup
    for _ in range(10):
        _ = fused_add_rms_norm(residual.clone(), x, weight)
        _ = residual_ref + x
        _ = F.rms_norm(_, _.shape[-1:], weight, eps=1e-6)
    torch.cuda.synchronize()
    
    # Fused
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = fused_add_rms_norm(residual.clone(), x, weight)
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / 1000 * 1000
    
    # 分离
    t0 = time.perf_counter()
    for _ in range(1000):
        tmp = residual_ref + x
        _ = F.rms_norm(tmp, tmp.shape[-1:], weight, eps=1e-6)
    torch.cuda.synchronize()
    t_sep = (time.perf_counter() - t0) / 1000 * 1000
    
    print(f"分离版本: {t_sep:.3f} ms")
    print(f"融合版本: {t_fused:.3f} ms")
    print(f"加速比: {t_sep/t_fused:.2f}x (省 {(t_sep-t_fused):.3f} ms)")