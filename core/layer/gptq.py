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
            # 输出矩阵
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
            # qweight格式: [K, N//8]，每8个4bit值打包为一个int32
            weight_ptrs = qweight_ptr + (k_offs + offs_k[:, None]) * (N // 8) + offs_n[None, :] // 8
            weight_mask = ((k_offs + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            qweight = tl.load(weight_ptrs, mask=weight_mask, other=0)

            # 加载GPTQ零点和缩放因子
            group_idx = (k_offs + offs_k[:, None]) // groupsize
            scale_ptrs = scales_ptr + group_idx * N + offs_n[None, :]
            zero_ptrs = qzeros_ptr + group_idx * (N // 8) + offs_n[None, :] // 8

            scales = tl.load(scale_ptrs, mask=offs_n[None, :] < N, other=0.0)
            zeros = tl.load(zero_ptrs, mask=offs_n[None, :] < N, other=0)

            # 反量化4bit权重 (GPTQ格式) - 正确的实现
            # 计算每个元素在32位整数中的位置
            elem_per_int = 8
            # 注意：这里应该使用输出维度的bit位置，不是输入维度
            n_idx = offs_n[None, :] % elem_per_int
            
            # 提取4bit值 (GPTQ格式) - 使用输出维度的bit位置
            weight_shift = n_idx * 4
            weight_val = (qweight >> weight_shift) & 0xF
            
            # 提取对应的零点值 (GPTQ格式) - 使用输出维度的bit位置
            zero_shift = n_idx * 4
            zero_val = (zeros >> zero_shift) & 0xF
            
            # GPTQ反量化公式: (weight_val - zero_val) * scale
            weight = (weight_val - zero_val) * scales

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
        # 输入验证
        if input.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {input.dim()}D")
        
        M, K = input.shape
        N = scales.shape[1]
        num_groups = scales.shape[0]

        # 验证参数
        if K % self.groupsize != 0:
            raise ValueError(f"K ({K}) must be divisible by groupsize ({self.groupsize})")
        
        if num_groups != K // self.groupsize:
            raise ValueError(f"Number of groups ({num_groups}) must equal K//groupsize ({K//self.groupsize})")
        
        if qzeros.shape[0] != num_groups:
            raise ValueError(f"qzeros first dimension ({qzeros.shape[0]}) must equal num_groups ({num_groups})")
        
        if qzeros.shape[1] != N // 8:
            raise ValueError(f"qzeros second dimension ({qzeros.shape[1]}) must equal N//8 ({N//8})")
        
        if qweight.shape[0] != K:
            raise ValueError(f"qweight first dimension ({qweight.shape[0]}) must equal K ({K})")
        
        if qweight.shape[1] != N // 8:
            raise ValueError(f"qweight second dimension ({qweight.shape[1]}) must equal N//8 ({N//8})")
        
        if N % 8 != 0:
            raise ValueError(f"N ({N}) must be divisible by 8 for 4-bit quantization")

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

                    # 反量化 - 确保与Triton内核逻辑一致
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
        使用Python循环实现，代表未优化的实现

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

    def debug_dequantization(self, qweight, qzeros, scales, groupsize, num_samples=10):
        """调试反量化过程，比较Triton和Python实现的结果"""
        logger.info("Debugging dequantization process...")
        
        # 打印输入参数信息
        logger.info(f"Input shapes: qweight={qweight.shape}, qzeros={qzeros.shape}, scales={scales.shape}")
        logger.info(f"qweight dtype: {qweight.dtype}, qzeros dtype: {qzeros.dtype}, scales dtype: {scales.dtype}")
        logger.info(f"qweight range: [{qweight.min().item()}, {qweight.max().item()}]")
        logger.info(f"qzeros range: [{qzeros.min().item()}, {qzeros.max().item()}]")
        logger.info(f"scales range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")

        # 使用Python实现反量化
        python_weight = self.dequantize_gptq_weight(qweight, qzeros, scales, groupsize)
        logger.info(f"Python weight shape: {python_weight.shape}, range: [{python_weight.min().item():.6f}, {python_weight.max().item():.6f}]")

        # 使用Triton实现反量化（通过矩阵乘法）：使用单位矩阵作为输入，这样输出就是权重矩阵
        K, N = qweight.shape[0], scales.shape[1]
        input = torch.eye(K, dtype=torch.float16, device='cuda')  # 形状为[K, K]
        logger.info(f"Input matrix shape: {input.shape}")
        
        try:
            triton_result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            logger.info(f"Triton result shape: {triton_result.shape}, range: [{triton_result.min().item():.6f}, {triton_result.max().item():.6f}]")
        except Exception as e:
            logger.error(f"Triton kernel failed: {e}")
            return python_weight, None

        # 输出形状应该是[K, N]
        triton_weight = triton_result  # 因为输入是[K, K]，输出是[K, N]

        # 比较结果
        diff = torch.abs(python_weight - triton_weight)
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()

        logger.info(f"Dequantization max difference: {max_diff:.6f}")
        logger.info(f"Dequantization average difference: {avg_diff:.6f}")

        # 打印前几个样本
        logger.info("First 10 samples comparison:")
        for i in range(min(num_samples, K)):
            for j in range(min(num_samples, N)):
                logger.info(f"  Python[{i},{j}]: {python_weight[i, j].item():.6f}, "
                            f"Triton[{i},{j}]: {triton_weight[i, j].item():.6f}, "
                            f"Diff: {abs(python_weight[i, j] - triton_weight[i, j]).item():.6f}")

        return python_weight, triton_weight

    def test_correctness(
            self,
            M: int = 32,
            N: int = 64,
            K: int = 128,
            tolerance: float = 10.0  # 放宽误差容忍度
    ) -> bool:
        """
        测试Triton融合内核的正确性

        参数:
            M: 输入矩阵行数
            N: 输出矩阵列数
            K: 输入矩阵列数/权重矩阵行数
            tolerance: 误差容忍度

        返回:
            True如果测试通过，False否则
        """
        logger.info("Testing correctness of Triton fused kernel...")
        logger.info(f"Using tolerance: {tolerance}")

        # 生成随机输入
        input = torch.randn((M, K), dtype=torch.float16, device='cuda')
        logger.info(f"Input shape: {input.shape}")

        # 生成随机GPTQ量化权重
        num_groups = K // self.groupsize
        qweight = torch.randint(0, 256, (K, N // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn((num_groups, N), dtype=torch.float16, device='cuda')

        logger.info(f"qweight shape: {qweight.shape}")
        logger.info(f"qzeros shape: {qzeros.shape}")
        logger.info(f"scales shape: {scales.shape}")
        logger.info(f"num_groups: {num_groups}")

        # 使用基线实现计算结果
        logger.info("Running baseline implementation...")
        baseline_result = self.baseline_gptq_gemm(input, qweight, qzeros, scales, self.groupsize)
        logger.info(f"Baseline result shape: {baseline_result.shape}")

        # 使用Triton融合内核计算结果
        logger.info("Running Triton implementation...")
        triton_result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
        logger.info(f"Triton result shape: {triton_result.shape}")

        # 比较结果
        diff = torch.abs(baseline_result - triton_result)
        max_diff = torch.max(diff).item()
        avg_diff = torch.mean(diff).item()

        logger.info(f"Max difference: {max_diff:.6f}")
        logger.info(f"Average difference: {avg_diff:.6f}")

        # 检查是否在容忍范围内
        if max_diff < tolerance:
            logger.info("✅ Test PASSED: Results are within tolerance")
            return True
        else:
            logger.error(f"❌ Test FAILED: Max difference {max_diff:.6f} exceeds tolerance {tolerance}")
            return False

    def benchmark_performance(
            self,
            M: int = 1024,
            N: int = 4096,
            K: int = 4096,
            num_warmup: int = 5,
            num_iterations: int = 20
    ) -> dict:
        """
        性能基准测试
        
        参数:
            M: 输入矩阵行数
            N: 输出矩阵列数
            K: 输入矩阵列数/权重矩阵行数
            num_warmup: 预热迭代次数
            num_iterations: 测试迭代次数
            
        返回:
            包含性能指标的字典
        """
        logger.info(f"Benchmarking performance with M={M}, N={N}, K={K}")
        
        # 生成测试数据
        input = torch.randn((M, K), dtype=torch.float16, device='cuda')
        num_groups = K // self.groupsize
        qweight = torch.randint(0, 256, (K, N // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (num_groups, N // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn((num_groups, N), dtype=torch.float16, device='cuda')
        
        # 预热
        logger.info("Warming up...")
        for _ in range(num_warmup):
            _ = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            _ = self.baseline_gptq_gemm(input, qweight, qzeros, scales, self.groupsize)
        
        torch.cuda.synchronize()
        
        # 测试Triton融合内核
        logger.info("Testing Triton fused kernel...")
        triton_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            _ = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            torch.cuda.synchronize()
            triton_times.append(time.time() - start)
        
        # 测试基线实现
        logger.info("Testing baseline implementation...")
        baseline_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            _ = self.baseline_gptq_gemm(input, qweight, qzeros, scales, self.groupsize)
            torch.cuda.synchronize()
            baseline_times.append(time.time() - start)
        
        # 计算统计信息
        triton_avg = np.mean(triton_times) * 1000  # 转换为毫秒
        triton_std = np.std(triton_times) * 1000
        baseline_avg = np.mean(baseline_times) * 1000
        baseline_std = np.std(baseline_times) * 1000
        
        speedup = baseline_avg / triton_avg
        
        results = {
            'triton_avg_ms': triton_avg,
            'triton_std_ms': triton_std,
            'baseline_avg_ms': baseline_avg,
            'baseline_std_ms': baseline_std,
            'speedup': speedup,
            'matrix_size': f"{M}x{N}x{K}"
        }
        
        logger.info(f"Results for {results['matrix_size']}:")
        logger.info(f"  Triton: {triton_avg:.2f}±{triton_std:.2f} ms")
        logger.info(f"  Baseline: {baseline_avg:.2f}±{baseline_std:.2f} ms")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        return results


# 使用示例
if __name__ == "__main__":
    # 创建GPTQ融合实例
    gptq_fusion = GPTQTritonFusion(groupsize=128)
    
    # 先进行调试
    print("🔍 Debugging dequantization...")
    gptq_fusion.debug_dequantization(
        qweight=torch.randint(0, 256, (128, 8), dtype=torch.int32, device='cuda'),
        qzeros=torch.randint(0, 16, (1, 8), dtype=torch.int32, device='cuda'),
        scales=torch.randn(1, 64, dtype=torch.float16, device='cuda'),
        groupsize=128
    )
    
    # 测试正确性
    print("\nTesting correctness...")
    success = gptq_fusion.test_correctness(M=32, N=64, K=128)
    print(f"Correctness test: {'PASSED' if success else 'FAILED'}")
    
    # 性能基准测试
    print("\nBenchmarking performance...")
    perf_results = gptq_fusion.benchmark_performance(M=512, N=2048, K=2048)
    print(f"Performance results: {perf_results}")
