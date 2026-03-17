import triton
import triton.language as tl
import torch
import time


# ==========================================
# Triton 3.6.0 正确且高效的Matmul
# 针对1x4096 × 4096x4096优化，L2缓存友好
# ==========================================
@triton.jit
def qwen_gemv_correct_kernel(
        a_ptr, b_ptr, c_ptr,
        N, K,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    # 每个线程块处理BLOCK_SIZE_N个输出元素
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    # 初始化累加器（FP32保证精度）
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # 遍历K维度，每次处理BLOCK_SIZE_K个元素
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        # ========== 修复1：正确的A矩阵加载（1xK，连续内存） ==========
        a_val = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0)

        # ========== 修复2：正确的B矩阵加载（KxN，行优先） ==========
        # B的索引：offs_k * N + offs_n（行优先存储）
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
        b_val = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # ========== 优化：利用Triton的自动dot product优化 ==========
        # 直接用tl.dot，让编译器自动生成Tensor Core指令
        accumulator += tl.dot(a_val, b_val)

    # 转换回FP16并存储结果
    c_val = accumulator.to(torch.float16)
    tl.store(c_ptr + offs_n, c_val, mask=mask_n)


def triton_qwen_matmul_correct(a, b):
    """
    针对Qwen-7B解码的Matmul: a(1x4096) @ b(4096x4096) -> c(1x4096)
    """
    # 确保输入是连续的FP16张量
    a = a.contiguous().to(torch.float16).flatten()  # 展平为1D向量
    b = b.contiguous().to(torch.float16)

    K = a.shape[0]
    K2, N = b.shape
    assert K == K2 == 4096, "仅支持4096维K"
    assert a.shape[0] == 4096, "A的形状必须是1x4096"

    # 初始化输出
    c = torch.empty((N,), device=a.device, dtype=torch.float16).contiguous()

    # ========== 优化：L2缓存友好的分块大小 ==========
    # BLOCK_SIZE_K=1024：每次处理1024个K元素，刚好放进L2
    # BLOCK_SIZE_N=128：每个Block处理128个输出元素
    BLOCK_SIZE_K = 1024
    BLOCK_SIZE_N = 128
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_blocks_n,)

    # 启动核函数
    qwen_gemv_correct_kernel[grid](
        a, b, c,
        N, K,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return c.reshape(1, N)  # 恢复为1xN形状


# ==========================================
# 性能测试与对比
# ==========================================
if __name__ == "__main__":
    # 检查CUDA可用性
    assert torch.cuda.is_available(), "需要CUDA环境"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Qwen-7B解码形状：1x4096 × 4096x4096
    M, K, N = 1, 4096, 4096
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    # 预热（避免首次编译开销）
    print("正在预热...")
    for _ in range(10):
        _ = a @ b
        _ = triton_qwen_matmul_correct(a, b)
    torch.cuda.synchronize()

    # 测试1: PyTorch原生Matmul (CUBLAS)
    iter_num = 1000
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iter_num):
        c_torch = a @ b
        torch.cuda.synchronize()
    torch_time = (time.time() - start) / iter_num * 1000

    # 测试2: Triton正确优化版Matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iter_num):
        c_triton = triton_qwen_matmul_correct(a, b)
        torch.cuda.synchronize()
    triton_time = (time.time() - start) / iter_num * 1000

    # 正确性验证
    max_error = torch.max(torch.abs(c_torch - c_triton)).item()
    mean_error = torch.mean(torch.abs(c_torch - c_triton)).item()

    # 输出结果
    print("\n" + "=" * 50)
    print(f"Qwen-7B解码形状: {a.shape} × {b.shape}")
    print(f"Triton版本: 3.6.0 (L2缓存优化版)")
    print("=" * 50)
    print(f"PyTorch原生 (CUBLAS):  {torch_time:.4f} ms/iter")
    print(f"Triton优化版:          {triton_time:.4f} ms/iter")
    print(f"性能提升:              {torch_time / triton_time:.2f}x")
    print("=" * 50)
    print(f"结果最大误差:          {max_error:.6f}")
    print(f"结果平均误差:          {mean_error:.6f}")
    print("=" * 50)
    print("验证状态:              " + ("✅ 通过" if max_error < 1e-3 else "❌ 失败"))