import triton
import triton.language as tl
import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Optional
import numpy as np

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 将反量化函数移到模块级别，使其成为全局函数
@triton.jit
def dequantize_gptq_4bit(qweight, zeros, scales, offs_k, offs_n):
    """反量化GPTQ 4bit权重"""
    # 计算每个元素在32位整数中的位置
    elem_per_int = 8
    k_idx = offs_k[:, None] % elem_per_int
    n_idx = offs_n[None, :]

    # 提取4bit值 (GPTQ格式)
    shift = k_idx * 4
    weight_val = (qweight >> shift) & 0xF

    # 提取对应的零点值 (GPTQ格式)
    zero_shift = (n_idx % 8) * 4
    zero_val = (zeros >> zero_shift) & 0xF

    # GPTQ反量化公式: (weight_val - zero_val) * scale
    dequantized = (weight_val - zero_val) * scales
    return dequantized


class GPTQTritonFusion:
    """GPTQ 4bit量化Triton融合内核测试类"""

    def __init__(self, groupsize=128):
        self.groupsize = groupsize

    @triton.jit
    def fused_gptq_gemm_kernel_4bit(
            # 输入矩阵A
            a_ptr, a_row_stride, a_col_stride,
            # GPTQ量化权重参数
            qweight_ptr, qzeros_ptr, scales_ptr,
            # 输出矩阵C
            c_ptr, c_row_stride, c_col_stride,
            # 矩阵维度
            M, N, K,
            # GPTQ量化参数
            groupsize: tl.constexpr,
            # 分块参数
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr
    ):
        """
        融合GPTQ 4bit反量化与矩阵乘法的优化内核
        """
        # 程序ID
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        # 创建偏移量
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # 初始化累加器
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # 主循环
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # 计算当前K块
            k_offs = k * BLOCK_SIZE_K

            # 加载A矩阵块
            a_ptrs = a_ptr + (offs_m[:, None] * a_row_stride +
                              (k_offs + offs_k[None, :]) * a_col_stride)
            a_mask = (offs_m[:, None] < M) & ((k_offs + offs_k[None, :]) < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # 加载量化权重并反量化 (GPTQ格式)
            weight_ptrs = qweight_ptr + ((k_offs + offs_k[:, None]) // 8 * (N // 8) +
                                         offs_n[None, :] // 8)
            weight_mask = ((k_offs + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            qweight = tl.load(weight_ptrs, mask=weight_mask, other=0)

            # 加载GPTQ零点和缩放因子
            group_idx = (k_offs + offs_k[:, None]) // groupsize
            scale_ptrs = scales_ptr + group_idx * N + offs_n[None, :]
            zero_ptrs = qzeros_ptr + group_idx * (N // 8) + offs_n[None, :] // 8

            scales = tl.load(scale_ptrs, mask=offs_n[None, :] < N, other=0.0)
            zeros = tl.load(zero_ptrs, mask=offs_n[None, :] < N, other=0)

            # 反量化4bit权重 (GPTQ格式)
            # 直接调用模块级别的函数，而不是通过类名调用
            weight = dequantize_gptq_4bit(qweight, zeros, scales, offs_k, offs_n)

            # 矩阵乘法累加
            accumulator += tl.dot(a, weight)

        # 存储结果
        c_ptrs = c_ptr + (offs_m[:, None] * c_row_stride +
                          offs_n[None, :] * c_col_stride)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

    def fused_gptq_gemm_4bit(
            self,
            input: torch.Tensor,
            qweight: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor,
            block_size_m: int = 64,
            block_size_n: int = 64,
            block_size_k: int = 64
    ) -> torch.Tensor:
        """
        融合GPTQ 4bit反量化与矩阵乘法的优化函数

        参数:
            input: 输入矩阵 [M, K]
            qweight: 量化权重 [K, N//8] (int32)
            qzeros: 零点值 [num_groups, N//8] (int32)
            scales: 缩放因子 [num_groups, N] (float16)
            block_size_m: M维度分块大小
            block_size_n: N维度分块大小
            block_size_k: K维度分块大小

        返回:
            输出矩阵 [M, N]
        """
        M, K = input.shape
        N = scales.shape[1]
        num_groups = scales.shape[0]

        # 验证参数
        assert num_groups == K // self.groupsize
        assert qzeros.shape[0] == num_groups
        assert qzeros.shape[1] == N // 8
        assert qweight.shape[0] == K
        assert qweight.shape[1] == N // 8

        # 分配输出张量
        result = torch.empty((M, N), dtype=input.dtype, device=input.device)

        # 计算网格大小
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        # 启动内核
        self.fused_gptq_gemm_kernel_4bit[grid](
            input, input.stride(0), input.stride(1),
            qweight, qzeros, scales,
            result, result.stride(0), result.stride(1),
            M, N, K,
            self.groupsize,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k
        )

        return result

    @staticmethod
    def dequantize_gptq_weight(
            qweight: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor,
            groupsize: int
    ) -> torch.Tensor:
        """
        反量化GPTQ权重（参考实现，用于验证）

        参数:
            qweight: 量化权重 [K, N//8] (int32)
            qzeros: 零点值 [num_groups, N//8] (int32)
            scales: 缩放因子 [num_groups, N] (float16)
            groupsize: 分组大小

        返回:
            反量化后的权重 [K, N] (float16)
        """
        K, _ = qweight.shape
        N = scales.shape[1]
        num_groups = scales.shape[0]

        # 反量化权重
        dequantized_weight = torch.zeros((K, N), dtype=torch.float16, device=qweight.device)

        for group_idx in range(num_groups):
            start_idx = group_idx * groupsize
            end_idx = min(start_idx + groupsize, K)

            for k in range(start_idx, end_idx):
                for n in range(N):
                    # 计算在qweight中的位置
                    weight_idx = k
                    byte_idx = n // 8
                    bit_shift = (n % 8) * 4

                    # 提取4bit权重值
                    weight_val = (qweight[weight_idx, byte_idx] >> bit_shift) & 0xF

                    # 提取零点值
                    zero_val = (qzeros[group_idx, byte_idx] >> bit_shift) & 0xF

                    # 提取缩放因子
                    scale_val = scales[group_idx, n]

                    # 反量化
                    dequantized_weight[k, n] = (weight_val - zero_val) * scale_val

        return dequantized_weight

    @staticmethod
    def baseline_gptq_gemm(
            input: torch.Tensor,
            qweight: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor,
            groupsize: int
    ) -> torch.Tensor:
        """
        基线实现：先反量化权重，再进行矩阵乘法

        参数:
            input: 输入矩阵 [M, K]
            qweight: 量化权重 [K, N//8] (int32)
            qzeros: 零点值 [num_groups, N//8] (int32)
            scales: 缩放因子 [num_groups, N] (float16)
            groupsize: 分组大小

        返回:
            输出矩阵 [M, N]
        """
        # 先反量化权重
        dequantized_weight = GPTQTritonFusion.dequantize_gptq_weight(
            qweight, qzeros, scales, groupsize
        )

        # 再进行矩阵乘法
        result = torch.matmul(input, dequantized_weight)

        return result

    def test_correctness(
            self,
            M: int = 32,
            N: int = 64,
            K: int = 128
    ) -> bool:
        """
        测试Triton融合内核的正确性

        参数:
            M: 输入矩阵行数
            N: 输出矩阵列数
            K: 输入矩阵列数/权重矩阵行数

        返回:
            True如果测试通过，False否则
        """
        logger.info("Testing correctness of Triton fused kernel...")

        # 生成随机输入
        input = torch.randn((M, K), dtype=torch.float16, device='cuda')

        # 生成随机GPTQ量化权重
        num_groups = K // self.groupsize
        qweight = torch.randint(0, 256, (K, N // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn((num_groups, N), dtype=torch.float16, device='cuda')

        # 使用基线实现计算结果
        baseline_result = self.baseline_gptq_gemm(input, qweight, qzeros, scales, self.groupsize)

        # 使用Triton融合内核计算结果
        triton_result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)

        # 比较结果
        diff = torch.abs(baseline_result - triton_result)
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()

        logger.info(f"Max difference: {max_diff:.6f}")
        logger.info(f"Average difference: {avg_diff:.6f}")

        # 允许有一定的数值误差
        tolerance = 1e-3
        if max_diff < tolerance:
            logger.info("✓ Correctness test passed!")
            return True
        else:
            logger.error("✗ Correctness test failed!")
            return False

    def test_performance(
            self,
            M: int = 256,
            N: int = 512,
            K: int = 1024,
            warmup: int = 10,
            repeats: int = 100
    ):
        """
        测试Triton融合内核的性能

        参数:
            M: 极简输入矩阵行数
            N: 输出矩阵列数
            K: 输入矩阵列数/权重矩阵行数
            warmup: 预热次数
            repeats: 测试次数
        """
        logger.info("Testing performance of Triton fused kernel...")

        # 生成随机输入
        input = torch.randn((M, K), dtype=torch.float16, device='cuda')

        # 生成随机GPTQ量化权重
        num_groups = K // self.groupsize
        qweight = torch.randint(0, 256, (K, N // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn((num_groups, N), dtype=torch.float16, device='极简')

        # 预热
        for _ in range(warmup):
            self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)

        # 测试Triton融合内核性能
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(repeats):
            triton_result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)

        torch.cuda.synchronize()
        triton_time = (time.time() - start极简) / repeats * 1000  # ms

        # 测试基线实现性能
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(repeats):
            baseline_result = self.baseline_gptq_gemm(input, qweight, qzeros, scales, self.groupsize)

        torch.cuda.synchronize()
        baseline_time = (time.time() - start_time) / repeats * 1000  # ms

        # 计算加速比
        speedup = baseline_time / triton_time

        logger.info(f"Baseline time: {baseline_time:.3f} ms")
        logger.info(f"Triton time: {triton_time:.3f} ms")
        logger.info(f"Speedup: {speedup:.2f}x")

        return triton_time, baseline_time, speedup

    def run_comprehensive_test(self):
        """运行全面的测试"""
        logger.info("Running comprehensive tests for GPTQ Triton fusion...")

        # 测试正确性
        correctness_passed = self.test_correctness()

        if not correctness_passed:
            logger.error("Correctness test failed! Aborting performance test.")
            return False

        # 测试性能
        logger.info("")
        performance_results = []

        # 测试不同矩阵大小
        test_cases = [
            (32, 64, 128),  # 小矩阵
            (256, 512, 1024),  # 中等矩阵
            (1024, 2048, 4096)  # 大矩阵
        ]

        for M, N, K in test_cases:
            logger.info(f"Testing performance for M={M}, N={N}, K={K}")
            triton_time, baseline_time, speedup = self.test_performance(M, N, K)
            performance_results.append({
                'M': M, 'N': N, 'K': K,
                'triton_time': triton_time,
                'baseline_time': baseline_time,
                'speedup': speedup
            })
            logger.info("")

        # 打印汇总结果
        logger.info("=== Test Results Summary ===")
        for result in performance_results:
            logger.info(f"M={result['M']}, N={result['N']}, K={result['K']}: "
                        f"Triton={result['triton_time']:.2f}ms, "
                        f"Baseline={result['baseline_time']:.2极简}ms, "
                        f"Speedup={result['speedup']:.2f}x")

        return True


def main():
    """主函数：运行测试"""
    logger.info("Starting GPTQ Triton fusion tests...")

    # 创建测试实例
    gptq_fusion = GPTQTritonFusion(groupsize=128)

    # 运行全面测试
    success = gptq_fusion.run_comprehensive_test()

    if success:
        logger.info("✓ All tests passed! Triton fusion is ready for integration.")
    else:
        logger.error("✗ Tests failed! Please check the implementation.")

    return success


if __name__ == "__main__":
    # 运行测试
    main()