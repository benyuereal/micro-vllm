"""
极限优化版 Triton GEMM - 专门针对 32×4096×4096 场景
不需要 autotune，直接使用最优配置

核心优化：
1. 硬编码最优 block 配置（针对 M=32 精确优化）
2. 充分利用 Tensor Core
3. 最优软件流水线参数
4. 最小化 kernel launch overhead
"""

import torch
import triton
import triton.language as tl
import time


# ============================================================================
# 针对 32×4096×4096 的最优配置（硬编码，无 autotune overhead）
# ============================================================================

# 分析：
# M = 32:
#   - BLOCK_M = 32 完美匹配，只需 1 个 block 沿 M 维度
#   - 最大化 SM 利用率
# N = 4096:
#   - BLOCK_N = 128: 4096/128 = 32 个 block 沿 N 维度
#   - 充分利用 Tensor Core
# K = 4096:
#   - BLOCK_K = 64: 4096/64 = 64 次迭代
#   - 平衡计算和内存访问

OPTIMAL_BLOCK_M = 32
OPTIMAL_BLOCK_N = 128
OPTIMAL_BLOCK_K = 64
OPTIMAL_NUM_STAGES = 4
OPTIMAL_NUM_WARPS = 8


@triton.jit
def matmul_kernel_extreme(
    # 指针
    a_ptr, b_ptr, c_ptr,
    # 维度
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    """
    极限优化版 GEMM Kernel

    专门针对 M=32, N=4096, K=4096 场景优化
    使用硬编码的 block 配置，避免 autotune overhead
    """

    # -----------------------------------------------------------
    # 配置（硬编码最优值）
    # -----------------------------------------------------------
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64

    # -----------------------------------------------------------
    # Super-grouping for L2 cache locality
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = M // BLOCK_M  # = 1 for M=32
    num_pid_n = N // BLOCK_N  # = 32 for N=4096

    # 由于 M=32，num_pid_m=1，super-grouping 简化为
    pid_m = 0
    pid_n = pid

    # -----------------------------------------------------------
    # 计算 block 位置
    # -----------------------------------------------------------
    # rm 范围: [0, 31]，完美覆盖 M=32
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask（实际上 M=32, N=4096 时不需要 mask，但保留以支持其他 shape）
    mask_m = rm < M
    mask_n = rn < N

    # -----------------------------------------------------------
    # 累加器（FP32 保证精度）
    # -----------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # K 维度迭代
    # K_tiles = 64 for K=4096, BLOCK_K=64
    # -----------------------------------------------------------
    # 预计算指针基础偏移
    # A: (M, K), A[rm, :] 的起始位置
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # 行优先
    )

    # B: (K, N), B[:, rn] 的起始位置
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),  # 对于 B 的 K 维度迭代
    )

    # 使用 block pointer 进行迭代
    for k in range(0, K, BLOCK_K):
        # 加载 A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 加载 B tile: [BLOCK_K, BLOCK_N]
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Tensor Core 矩阵乘法
        acc += tl.dot(a, b, allow_tf32=False)

        # 更新 block pointer
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # -----------------------------------------------------------
    # 存储结果
    # -----------------------------------------------------------
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# ============================================================================
# 使用 pipe-based 软件流水线的版本（更激进）
# ============================================================================

@triton.jit
def matmul_kernel_pipelined(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    """
    使用显式软件流水线的 GEMM Kernel

    通过 tl.dot 加上 num_stages 编译器提示实现流水线
    """

    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64

    pid = tl.program_id(axis=0)

    rm = tl.arange(0, BLOCK_M)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 基础指针
    a_base = a_ptr + rm[:, None] * stride_am
    b_base = b_ptr + rn[None, :] * stride_bn

    # 预取第一个 tile
    k_offset = tl.arange(0, BLOCK_K)
    a_ptrs = a_base + k_offset[None, :] * stride_ak
    b_ptrs = b_base + k_offset[:, None] * stride_bk

    # 流水线化的主循环
    K_tiles = K // BLOCK_K

    for k in tl.range(K_tiles):
        # 加载当前 tile
        k_start = k * BLOCK_K
        k_offset = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(a_base + k_offset[None, :] * stride_ak)
        b = tl.load(b_base + k_offset[:, None] * stride_bk)

        # 累加
        acc += tl.dot(a, b)

    # 存储
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


# ============================================================================
# 针对 M=32 特殊优化的最小 kernel
# ============================================================================

@triton.jit
def matmul_kernel_m32_optimized(
    a_ptr, b_ptr, c_ptr,
    N, K,  # M=32 硬编码
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    """
    专门为 M=32 优化的最小 kernel

    M=32 硬编码，消除运行时判断
    """

    BLOCK_M: tl.constexpr = 32  # 精确匹配 M=32
    BLOCK_N: tl.constexpr = 128
    BLOCK_K: tl.constexpr = 64

    pid = tl.program_id(axis=0)

    # rm 固定为 [0, 31]，不需要 mask
    rm = tl.arange(0, 32)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    # N 维度 mask
    mask_n = rn < N

    acc = tl.zeros((32, BLOCK_N), dtype=tl.float32)

    # K 维度循环
    for k in range(0, K, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)

        # 加载 A[0:32, k:k+64]
        a = tl.load(a_ptr + rm[:, None] * stride_am + k_offset[None, :] * stride_ak)

        # 加载 B[k:k+64, rn]
        b = tl.load(b_ptr + k_offset[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=mask_n[None, :], other=0.0)

        acc += tl.dot(a, b)

    # 存储 C[0:32, rn]
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(tl.float16), mask=mask_n[None, :])


def triton_matmul_extreme(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    极限优化版 Triton GEMM 接口
    """
    M, K = a.shape
    K2, N = b.shape

    assert M == 32, f"此 kernel 专为 M=32 优化，当前 M={M}"
    assert K == K2, f"维度不匹配: A.shape[1]={K} != B.shape[0]={K2}"

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Grid: M=32/BLOCK_M=1, N/BLOCK_N 个 block
    grid = (N // 128 + (N % 128 > 0),)

    # 使用 pipelined 版本
    matmul_kernel_pipelined[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_stages=OPTIMAL_NUM_STAGES,
        num_warps=OPTIMAL_NUM_WARPS,
    )

    return c


import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_v3(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -----------------------------------------------------------
    # ID 计算：直接线性映射，减少指令数
    pid = tl.program_id(axis=0)
    num_pid_n = N // BLOCK_N
    pid_m = 0  # 因为 M=32，只有一个 BLOCK_M
    pid_n = pid

    # -----------------------------------------------------------
    # 指针计算：回归最原始、最高效的加法偏移
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # 预计算初始指针
    # A 只需要在 K 维度移动，M 维度不动
    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    # B 只需要在 K 维度移动，N 维度不动
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # 核心循环
    for k in range(0, K, BLOCK_K):
        # 4096 对齐，不使用 mask 以获得极限性能
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # 核心计算
        acc += tl.dot(a, b)

        # 移动指针
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # -----------------------------------------------------------
    # 写回结果
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16))


def triton_matmul_v3(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # --- 针对 A100/H100 M=32 专门调优的参数 ---
    BLOCK_M = 32
    BLOCK_N = 64  # 从 128 缩小到 64，增加并行 Block 数到 64 个
    BLOCK_K = 128  # 增大 K Tile，让计算更密集
    num_warps = 4  # 减少 Warp 数量，提高单核效率
    num_stages = 4  # 增加流水线深度，掩盖加载 B 的延迟

    grid = (N // BLOCK_N,)

    matmul_kernel_v3[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# ============================================================================
# 性能测试
# ============================================================================

def benchmark_extreme():
    """
    极限优化版性能测试
    """
    M, N, K = 32, 4096, 4096

    print(f"\n{'='*70}")
    print(f"  极限优化版测试: M={M}, N={N}, K={K}")
    print(f"{'='*70}")

    # 创建数据
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')

    flops = 2.0 * M * N * K

    warmup = 20
    repeat = 200

    # -----------------------------------------------------------
    # cuBLAS
    # -----------------------------------------------------------
    for _ in range(warmup):
        c_cublas = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        c_cublas = torch.matmul(a, b)
    torch.cuda.synchronize()
    cublas_time = (time.perf_counter() - start) / repeat * 1000

    print(f"\n[cuBLAS]")
    print(f"  延迟: {cublas_time:.4f} ms")
    print(f"  性能: {flops / cublas_time / 1e9:.2f} TFLOPS")

    # -----------------------------------------------------------
    # Triton 极限优化版
    # -----------------------------------------------------------
    for _ in range(warmup):
        c_triton = triton_matmul_v3(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        c_triton = triton_matmul_v3(a, b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / repeat * 1000

    print(f"\n[Triton 极限优化版]")
    print(f"  延迟: {triton_time:.4f} ms")
    print(f"  性能: {flops / triton_time / 1e9:.2f} TFLOPS")
    print(f"  vs cuBLAS: {triton_time / cublas_time * 100:.2f}%")

    # -----------------------------------------------------------
    # 正确性验证
    # -----------------------------------------------------------
    diff = (c_triton.float() - c_cublas.float()).abs()
    max_diff = diff.max().item()
    print(f"\n[正确性] 最大差异: {max_diff:.6f}")

    # -----------------------------------------------------------
    # 结论
    # -----------------------------------------------------------
    ratio = triton_time / cublas_time * 100
    print(f"\n{'='*70}")
    if ratio < 100:
        print(f"🎉 超越 cuBLAS {100-ratio:.2f}%！")
    elif ratio < 105:
        print(f"✓ 达到 cuBLAS 的 {100/ratio*100:.1f}%")
    else:
        print(f"⚠ 为 cuBLAS 的 {100/ratio*100:.1f}%")
    print(f"{'='*70}")

    return cublas_time, triton_time


# ============================================================================
# 详细配置分析
# ============================================================================

def analyze_optimal_config():
    """
    分析为什么选择这个配置
    """
    print("\n" + "="*70)
    print("  最优配置分析 (M=32, N=4096, K=4096)")
    print("="*70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  参数              │  值    │  分析                                    │
├─────────────────────────────────────────────────────────────────────┤
│  BLOCK_M          │  32    │  精确匹配 M=32，只需 1 个 block 沿 M    │
│  BLOCK_N          │  128   │  N/128=32 个 block，充分利用 Tensor Core│
│  BLOCK_K          │  64    │  K/64=64 次迭代，平衡计算与访存         │
│  num_stages       │  4     │  4 级软件流水线隐藏内存延迟              │
│  num_warps        │  8     │  8 个 warp (256 threads) 平衡资源        │
├─────────────────────────────────────────────────────────────────────┤
│  Shared Memory 使用估算                                              │
│  A tile: 32×64×2 = 4KB (FP16)                                        │
│  B tile: 64×128×2 = 16KB (FP16)                                      │
│  Total per stage: 20KB × 4 stages = 80KB                             │
│  A100 共享内存: 164KB/SM，足够容纳                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Tensor Core 利用                                                    │
│  HMMA.16816 指令: 16×16×16 tile                                      │
│  BLOCK_M=32 = 2×16, BLOCK_N=128 = 8×16, BLOCK_K=64 = 4×16           │
│  每个 tile 完美对齐 Tensor Core 要求                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Grid 配置                                                           │
│  Grid = (32,) -- 32 个 block 沿 N 维度                               │
│  A100 有 108 个 SM，32 个 block 可完全并行执行                        │
└─────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误：需要 CUDA 设备")
        exit(1)

    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")

    analyze_optimal_config()
    benchmark_extreme()