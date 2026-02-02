import torch
import triton
import triton.language as tl


@triton.jit
def ln_fwd(
    X, Y, W,          # input, output, weight
    M, N,             # rows, cols (feature dim)
    stride_x, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    X_row = X + row * stride_x
    Y_row = Y + row * stride_x
    
    # 标量累加：平方和
    rms_sq = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        rms_sq += tl.sum(x * x)
    
    rstd = 1.0 / tl.sqrt(rms_sq / N + eps)
    
    # 归一化 + 仿射变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = x * rstd * w
        tl.store(Y_row + cols, y, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    RMSNorm 前向 (支持任意形状)
    x: (..., N), weight: (N,)
    """
    original_shape = x.shape
    x_flat = x.view(-1, x.size(-1))
    M, N = x_flat.shape
    
    y_flat = torch.empty_like(x_flat)  # 与输入同类型（如float16）
    BLOCK_SIZE = min(triton.next_power_of_2(N), 2048)
    
    ln_fwd[(M,)](
        x_flat, y_flat, weight,
        M, N,
        x_flat.stride(0), eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 512 else 8,
    )
    
    return y_flat.view(original_shape)


# === 修复后的验证 ===
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((4, 1024), device='cuda', dtype=torch.float16)
    w = torch.ones((1024,), device='cuda', dtype=torch.float16)
    
    y_triton = rms_norm(x, w, eps=1e-6)
    
    # PyTorch 参考实现（保持float32计算，最后转为float16比较）
    variance = x.float().pow(2).mean(-1, keepdim=True)
    y_ref = x.float() * torch.rsqrt(variance + 1e-6) * w.float()
    
    # 关键修复：将Triton输出转为float32再比较（避免类型错误）
    max_err = torch.max(torch.abs(y_triton.float() - y_ref)).item()
    is_close = torch.allclose(y_triton.float(), y_ref, rtol=1e-3, atol=1e-5)
    
    print(f"最大误差: {max_err:.2e}")
    print("✅ 通过" if is_close else "❌ 失败")