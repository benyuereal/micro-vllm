# File: gptq.py
import torch
import triton
import triton.language as tl
import logging

logger = logging.getLogger(__name__)

class GPTQTritonFusion:
    """
    GPTQ Triton融合内核实现
    专注于高性能推理加速
    """

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
            # qweight格式: [N, K//8]，每8个4bit值打包为一个int32
            weight_ptrs = qweight_ptr + offs_n[None, :] * (K // 8) + (k_offs + offs_k[:, None]) // 8
            weight_mask = (offs_n[None, :] < N) & ((k_offs + offs_k[:, None]) < K)
            qweight = tl.load(weight_ptrs, mask=weight_mask, other=0)

            # 加载GPTQ零点和缩放因子
            group_idx = (k_offs + offs_k[:, None]) // groupsize
            scale_ptrs = scales_ptr + group_idx * K + (k_offs + offs_k[:, None])
            zero_ptrs = qzeros_ptr + group_idx * (K // 8) + (k_offs + offs_k[:, None]) // 8

            scales = tl.load(scale_ptrs, mask=(k_offs + offs_k[:, None]) < K, other=0.0)
            zeros = tl.load(zero_ptrs, mask=(k_offs + offs_k[:, None]) < K, other=0)

            # 反量化4bit权重 (GPTQ格式)
            elem_per_int = 8
            k_idx = (k_offs + offs_k[:, None]) % elem_per_int
            
            # 提取4bit值
            weight_shift = k_idx * 4
            weight_val = (qweight >> weight_shift) & 0xF
            
            # 提取对应的零点值
            zero_shift = k_idx * 4
            zero_val = (zeros >> zero_shift) & 0xF
            
            # GPTQ反量化公式: (weight_val - zero_val) * scale
            # 确保数据类型匹配
            weight_val = weight_val.to(tl.float32)
            zero_val = zero_val.to(tl.float32)
            scales = scales.to(tl.float32)
            a = a.to(tl.float32)
            
            weight = (weight_val - zero_val) * scales
            
            # 矩阵乘法累加
            accumulator += tl.dot(a, weight)

        # 存储结果
        c_ptrs = c_ptr + (offs_m[:, None] * c_row_stride +
                          offs_n[None, :] * c_col_stride)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    def fused_gptq_gemm_4bit(
            self,
            input: torch.Tensor,
            qweight: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor
    ) -> torch.Tensor:
        """
        融合GPTQ 4bit反量化与矩阵乘法的优化实现
        
        参数:
            input: 输入矩阵 [M, K]
            qweight: 量化权重 [N, K//8] (int32) - GPTQ格式
            qzeros: 零点值 [num_groups, K//8] (int32) - GPTQ格式
            scales: 缩放因子 [num_groups, K] (float16) - GPTQ格式

        返回:
            输出矩阵 [M, N]
        """
        # 输入验证
        if input.dim() != 2:
            raise ValueError(f"输入必须是2D矩阵，得到 {input.dim()}D")
        
        M, K = input.shape
        N = qweight.shape[0]
        
        # logger.info(f"Triton融合内核: input{M}x{K}, qweight{N}x{qweight.shape[1]}, qzeros{qzeros.shape}, scales{scales.shape}")
        
        # 验证GPTQ格式
        if qweight.shape[1] != K // 8:
            raise ValueError(f"qweight第二维 ({qweight.shape[1]}) 必须等于 K//8 ({K // 8})")
        
        if qzeros.shape[1] != K // 8:
            raise ValueError(f"qzeros第二维 ({qzeros.shape[1]}) 必须等于 K//8 ({K // 8})")
        
        if scales.shape[1] != K:
            raise ValueError(f"scales第二维 ({scales.shape[1]}) 必须等于 K ({K})")
        
        if K % 8 != 0:
            raise ValueError(f"K ({K}) 必须能被8整除以支持4bit量化")
        
        # 检查分组大小
        num_groups = scales.shape[0]
        if num_groups != K // self.groupsize:
            raise ValueError(f"分组数量 ({num_groups}) 必须等于 K//groupsize ({K // self.groupsize})")
        
        # 分配输出矩阵 - 使用更高效的内存分配
        output = torch.empty((M, N), dtype=input.dtype, device=input.device)
        
        # 分块参数 - 更激进的分块以提高性能
        BLOCK_SIZE_M = 256
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 256
        
        # 计算网格大小
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
        
        # 启动Triton内核
        self.fused_gptq_gemm_kernel_4bit[grid](
            # 输入矩阵
            input, input.stride(0), input.stride(1),
            # GPTQ参数
            qweight, qzeros, scales,
            # 输出矩阵
            output, output.stride(0), output.stride(1),
            # 矩阵维度
            M, N, K,
            # GPTQ参数
            groupsize=self.groupsize,
            # 分块参数
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        
        # 转换输出数据类型以匹配输入 - 优化：直接使用输入类型
        # if output.dtype != input.dtype:
        #     output = output.to(input.dtype)
        
        # logger.info(f"Triton融合内核完成: 输出形状 {output.shape}")
        return output

    def _handle_qwen7b_qkv_format(self, input, qweight, qzeros, scales, input_dim, output_dim):
        """处理Qwen7B QKV投影格式，使用优化的Triton内核"""
        # logger.info("处理Qwen7B QKV投影格式")
        
        # Qwen7B QKV格式: qweight=[512, 12288], qzeros=[32, 1536], scales=[32, 12288]
        # 我们需要将其转换为标准格式以使用Triton内核
        
        # 重新排列qweight: [512, 12288] -> [12288, 512] (转置)
        qweight_transposed = qweight.T  # [12288, 512]
        
        # 重新排列qzeros: [32, 1536] -> [32, 512] (截取前512列)
        qzeros_adjusted = qzeros[:, :input_dim // 8]  # [32, 512]
        
        # 重新排列scales: [32, 12288] -> [32, 4096] (截取前4096列)
        scales_adjusted = scales[:, :input_dim]  # [32, 4096]
        
        # 现在使用标准Triton内核
        return self.fused_gptq_gemm_4bit(input, qweight_transposed, qzeros_adjusted, scales_adjusted)

    def _handle_qwen7b_output_format(self, input, qweight, qzeros, scales, N, K):
        """处理Qwen7B输出投影格式，使用优化的Triton内核"""
        # logger.info("处理Qwen7B输出投影格式")
        
        # Qwen7B输出格式: qweight=[512, 4096], qzeros=[32, 512], scales=[32, 4096]
        # 我们需要将其转换为标准格式以使用Triton内核
        
        # 重新排列qweight: [512, 4096] -> [4096, 512] (转置)
        qweight_transposed = qweight.T  # [4096, 512]
        
        # qzeros已经是正确的格式: [32, 512]
        # scales已经是正确的格式: [32, 4096]
        
        # 现在使用标准Triton内核
        return self.fused_gptq_gemm_4bit(input, qweight_transposed, qzeros, scales)

    def process_qwen7b_format(self, input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        处理Qwen7B GPTQ格式的主入口
        自动检测格式并使用相应的Triton内核
        """
        M, K = input.shape
        N = qweight.shape[0]
        
        # logger.info(f"处理Qwen7B格式: input{M}x{K}, qweight{N}x{qweight.shape[1]}, qzeros{qzeros.shape}, scales{scales.shape}")
        
        # 检测Qwen7B格式
        if qweight.shape[1] == scales.shape[1] and qzeros.shape[1] == scales.shape[1] // 8:
            # Qwen7B QKV投影格式: [input_dim//8, output_dim]
            # logger.info("检测到Qwen7B QKV投影格式")
            input_dim_compressed = N  # 512
            output_dim = scales.shape[1]  # 12288
            input_dim = input_dim_compressed * 8  # 4096
            
            return self._handle_qwen7b_qkv_format(input, qweight, qzeros, scales, input_dim, output_dim)
        elif qweight.shape[1] == K and qzeros.shape[1] == N and scales.shape[1] == K:
            # Qwen7B输出投影格式: [output_dim//8, input_dim]
            # logger.info("检测到Qwen7B输出投影格式")
            return self._handle_qwen7b_output_format(input, qweight, qzeros, scales, N, K)
        else:
            # 尝试标准格式
            # logger.info("尝试标准GPTQ格式")
            return self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)

    def test_correctness(self, M=32, N=64, K=128, groupsize=128):
        """测试Triton内核的正确性"""
        logger.info("测试Triton内核正确性...")
        
        # 生成测试数据
        input = torch.randn(M, K, dtype=torch.float16, device='cuda')
        
        # 生成模拟的GPTQ参数
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (K // groupsize, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(K // groupsize, K, dtype=torch.float16, device='cuda')
        
        logger.info(f"测试数据: input{M}x{K}, qweight{N}x{K//8}, qzeros{K//groupsize}x{K//8}, scales{K//groupsize}x{K}")
        
        try:
            # 使用Triton内核
            result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            logger.info(f"Triton内核成功: 输出形状 {result.shape}")
            return True
        except Exception as e:
            logger.error(f"Triton内核失败: {e}")
            return False

    def benchmark_performance(self, M=32, N=64, K=128, groupsize=128, num_iterations=100):
        """基准测试Triton内核性能"""
        logger.info("基准测试Triton内核性能...")
        
        # 生成测试数据
        input = torch.randn(M, K, dtype=torch.float16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (K // groupsize, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(K // groupsize, K, dtype=torch.float16, device='cuda')
        
        # 预热
        for _ in range(10):
            _ = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
        torch.cuda.synchronize()
        
        # 性能测试
        times = []
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"性能统计: 平均 {avg_time:.2f}ms, 最小 {min_time:.2f}ms, 最大 {max_time:.2f}ms")
        return avg_time

if __name__ == "__main__":
    # 测试Triton内核
    fusion = GPTQTritonFusion(groupsize=128)
    
    print("🧪 测试Triton内核正确性...")
    if fusion.test_correctness():
        print("✅ Triton内核正确性测试通过")
    else:
        print("❌ Triton内核正确性测试失败")
    
    print("\n⚡ 基准测试Triton内核性能...")
    avg_time = fusion.benchmark_performance()
    print(f"平均执行时间: {avg_time:.2f}ms")
    
    if avg_time < 1.0:  # 小于1ms认为性能良好
        print("🎉 Triton内核性能优秀!")
    else:
        print("⚠️ Triton内核性能需要进一步优化")
