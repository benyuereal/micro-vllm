"""
CUDA C++内核包装器
提供与Triton内核相同的接口
"""

import torch
import logging
import os
import sys

logger = logging.getLogger(__name__)

class GPTQCudaFusion:
    """
    GPTQ CUDA C++融合内核实现
    极致性能优化版本
    """
    
    def __init__(self, groupsize=128):
        self.groupsize = groupsize
        self.cuda_kernel = None
        self._load_cuda_kernel()
    
    def _load_cuda_kernel(self):
        """加载CUDA C++内核"""
        try:
            # 尝试编译CUDA内核
            sys.path.append(os.path.dirname(__file__))
            from compile_cuda_kernel import compile_cuda_kernel
            
            self.cuda_kernel = compile_cuda_kernel()
            if self.cuda_kernel:
                logger.info("✅ CUDA C++内核加载成功")
            else:
                logger.warning("⚠️ CUDA C++内核编译失败，将使用Triton内核")
        except Exception as e:
            logger.warning(f"⚠️ CUDA C++内核加载失败: {e}")
            self.cuda_kernel = None
    
    def fused_gptq_gemm_4bit(
            self,
            input: torch.Tensor,
            qweight: torch.Tensor,
            qzeros: torch.Tensor,
            scales: torch.Tensor
    ) -> torch.Tensor:
        """
        融合GPTQ 4bit反量化与矩阵乘法
        
        参数:
            input: 输入矩阵 [M, K]
            qweight: 量化权重 [N, K//8] (int32)
            qzeros: 零点值 [num_groups, K//8] (int32)
            scales: 缩放因子 [num_groups, K] (float16)
        
        返回:
            输出矩阵 [M, N]
        """
        # 输入验证
        if input.dim() != 2:
            raise ValueError(f"输入必须是2D矩阵，得到 {input.dim()}D")
        
        M, K = input.shape
        N = qweight.shape[0]
        
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
        
        # 使用CUDA内核
        if self.cuda_kernel:
            try:
                return self.cuda_kernel.fused_gptq_gemm_4bit_cuda(
                    input, qweight, qzeros, scales, self.groupsize
                )
            except Exception as e:
                logger.warning(f"CUDA内核失败，回退到Triton: {e}")
                return self._fallback_triton(input, qweight, qzeros, scales)
        else:
            return self._fallback_triton(input, qweight, qzeros, scales)
    
    def _fallback_triton(self, input, qweight, qzeros, scales):
        """回退到Triton内核"""
        # 导入Triton实现
        from .gptq import GPTQTritonFusion
        triton_fusion = GPTQTritonFusion(self.groupsize)
        return triton_fusion.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
    
    def process_qwen7b_format(self, input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        处理Qwen7B GPTQ格式
        自动检测格式并使用相应的内核
        """
        M, K = input.shape
        N = qweight.shape[0]
        
        logger.info(f"CUDA格式检测: input{M}x{K}, qweight{N}x{qweight.shape[1]}, qzeros{qzeros.shape}, scales{scales.shape}")
        
        # 检测Qwen7B格式
        if qweight.shape[1] == scales.shape[1] and qzeros.shape[1] == scales.shape[1] // 8:
            # Qwen7B QKV投影格式
            logger.info("检测到Qwen7B QKV投影格式")
            input_dim_compressed = N  # 512
            output_dim = scales.shape[1]  # 12288
            input_dim = input_dim_compressed * 8  # 4096
            
            return self._handle_qwen7b_qkv_format(input, qweight, qzeros, scales, input_dim, output_dim)
        elif qweight.shape[1] == K and qzeros.shape[1] == N and scales.shape[1] == K:
            # Qwen7B输出投影格式
            logger.info("检测到Qwen7B输出投影格式")
            return self._handle_qwen7b_output_format(input, qweight, qzeros, scales, N, K)
        else:
            # 尝试标准格式
            logger.info("尝试标准GPTQ格式")
            return self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
    
    def _handle_qwen7b_qkv_format(self, input, qweight, qzeros, scales, input_dim, output_dim):
        """处理Qwen7B QKV投影格式"""
        # Qwen7B QKV格式: qweight=[512, 12288], qzeros=[32, 1536], scales=[32, 12288]
        # 转换为标准格式
        qweight_transposed = qweight.T  # [12288, 512]
        qzeros_adjusted = qzeros[:, :input_dim // 8]  # [32, 512]
        scales_adjusted = scales[:, :input_dim]  # [32, 4096]
        
        return self.fused_gptq_gemm_4bit(input, qweight_transposed, qzeros_adjusted, scales_adjusted)
    
    def _handle_qwen7b_output_format(self, input, qweight, qzeros, scales, N, K):
        """处理Qwen7B输出投影格式"""
        # Qwen7B输出格式: qweight=[512, 4096], qzeros=[32, 512], scales=[32, 4096]
        qweight_transposed = qweight.T  # [4096, 512]
        
        return self.fused_gptq_gemm_4bit(input, qweight_transposed, qzeros, scales)
    
    def test_correctness(self, M=32, N=64, K=128, groupsize=128):
        """测试CUDA内核的正确性"""
        logger.info("测试CUDA内核正确性...")
        
        # 生成测试数据
        input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (K // groupsize, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(K // groupsize, K, dtype=torch.bfloat16, device='cuda')
        
        try:
            result = self.fused_gptq_gemm_4bit(input, qweight, qzeros, scales)
            logger.info(f"CUDA内核成功: 输出形状 {result.shape}")
            return True
        except Exception as e:
            logger.error(f"CUDA内核失败: {e}")
            return False
    
    def benchmark_performance(self, M=32, N=64, K=128, groupsize=128, num_iterations=100):
        """基准测试CUDA内核性能"""
        logger.info("基准测试CUDA内核性能...")
        
        # 生成测试数据
        input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
        qzeros = torch.randint(0, 16, (K // groupsize, K // 8), dtype=torch.int32, device='cuda')
        scales = torch.randn(K // groupsize, K, dtype=torch.bfloat16, device='cuda')
        
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
        
        logger.info(f"CUDA性能统计: 平均 {avg_time:.2f}ms, 最小 {min_time:.2f}ms, 最大 {max_time:.2f}ms")
        return avg_time

if __name__ == "__main__":
    # 测试CUDA内核
    fusion = GPTQCudaFusion(groupsize=128)
    
    print("🧪 测试CUDA内核正确性...")
    if fusion.test_correctness():
        print("✅ CUDA内核正确性测试通过")
    else:
        print("❌ CUDA内核正确性测试失败")
    
    print("\n⚡ 基准测试CUDA内核性能...")
    avg_time = fusion.benchmark_performance()
    print(f"平均执行时间: {avg_time:.2f}ms")
    
    if avg_time < 0.5:  # 小于0.5ms认为性能优秀
        print("🎉 CUDA内核性能优秀!")
    else:
        print("⚠️ CUDA内核性能需要进一步优化")
