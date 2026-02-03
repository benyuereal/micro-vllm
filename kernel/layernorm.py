import torch
import triton
import triton.language as tl



@triton.jit
def ln_fwd(
    X, Y, W, B,           # 输入、输出、权重、偏置
    M, N,                 # 行数、列数（特征维）
    stride, eps,          # 行步长、epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    X_row = X + row * stride
    Y_row = Y + row * stride

    # --- 1. 循环计算均值 ---
    mean_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.).to(tl.float32)
        mean_acc += tl.where(mask, x, 0.)
    mean = tl.sum(mean_acc) / N

    # --- 2. 循环计算方差 ---
    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.).to(tl.float32)
        x_centered = tl.where(mask, x - mean, 0.)
        var_acc += x_centered * x_centered
    rstd = 1 / tl.sqrt(tl.sum(var_acc) / N + eps)  # 倒数标准差

    # --- 3. 归一化并仿射变换 ---
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + cols, y, mask=mask)


def layer_norm(x, weight, bias, eps=1e-5, BLOCK_SIZE=1024):
    """
    通用 LayerNorm 前向
    x: [M, N], weight/bias: [N]
    """
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    
    ln_fwd[grid](
        x, y, weight, bias,
        M, N,
        x.stride(0), eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y


if __name__ == "__main__":
    # 测试多种形状
    test_cases = [(2, 64), (1, 1024), (4, 4096), (8, 768)]
    eps = 1e-5

    for M, N in test_cases:
        print(f"\n>>> 测试形状 [M={M}, N={N}]")
        x = torch.randn((M, N), device='cuda')
        w = torch.randn((N,), device='cuda')
        b = torch.randn((N,), device='cuda')

        y_custom = layer_norm(x, w, b, eps)
        y_ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps)

        max_err = torch.max(torch.abs(y_custom - y_ref)).item()
        is_ok = torch.allclose(y_custom, y_ref, rtol=1e-3, atol=1e-5)
        print(f"  最大误差: {max_err:.2e} | 结果: {'✅' if is_ok else '❌'}")