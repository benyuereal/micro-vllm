import torch
import ctypes
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class FusedLNQKVGPTQ:
    """融合LayerNorm + QKV投影的GPTQ量化算子"""
    
    def __init__(self, kernel_path: Optional[str] = None):
        """
        初始化融合算子
        
        Args:
            kernel_path: CUDA内核库路径
        """
        self.kernel_path = kernel_path or self._find_kernel_path()
        self._cuda_kernel = None
        self._load_kernel()
    
    def _find_kernel_path(self) -> str:
        """查找CUDA内核库路径"""
        # 查找编译后的内核库
        possible_paths = [
            "cuda/ln_qkv_fusion_kernel.so",
            "cuda/ln_qkv_fusion_kernel.dll",
            "ln_qkv_fusion_kernel.so",
            "ln_qkv_fusion_kernel.dll"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"找不到CUDA内核库，尝试路径: {possible_paths}")
    
    def _load_kernel(self):
        """加载CUDA内核库"""
        try:
            self._cuda_kernel = ctypes.CDLL(self.kernel_path)
            logger.info(f"✅ 成功加载融合LN+QKV内核: {self.kernel_path}")
        except Exception as e:
            logger.error(f"❌ 加载CUDA内核失败: {e}")
            raise RuntimeError(f"无法加载CUDA内核: {e}")
    
    def fused_ln_qkv_gptq(
        self,
        input: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        num_heads: int,
        kv_num_heads: int,
        head_size: int,
        groupsize: int = 128,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        融合LayerNorm + QKV投影
        
        Args:
            input: 输入张量 [batch_size, seq_len, hidden_dim]
            qweight: GPTQ量化权重 [K//8, N]
            qzeros: GPTQ量化零点 [num_groups, groupsize//8]
            scales: GPTQ量化缩放 [num_groups, N]
            ln_weight: LayerNorm权重 [hidden_dim]
            ln_bias: LayerNorm偏置 [hidden_dim]
            num_heads: 注意力头数量
            kv_num_heads: KV头数量
            head_size: 头大小
            groupsize: GPTQ组大小
            eps: LayerNorm epsilon
            
        Returns:
            Tuple[Q, K, V]: QKV张量
        """
        # 验证输入形状
        batch_size, seq_len, hidden_dim = input.shape
        
        # 确保数据类型正确
        if input.dtype != torch.float16:
            input = input.to(torch.float16)
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        if ln_weight.dtype != torch.float16:
            ln_weight = ln_weight.to(torch.float16)
        if ln_bias.dtype != torch.float16:
            ln_bias = ln_bias.to(torch.float16)
        
        # 确保权重数据类型正确
        if qweight.dtype != torch.uint32:
            qweight = qweight.to(torch.uint32)
        if qzeros.dtype != torch.uint32:
            qzeros = qzeros.to(torch.uint32)
        
        # 创建输出张量
        q_output = torch.zeros(batch_size, num_heads, seq_len, head_size, 
                              dtype=torch.float16, device=input.device)
        k_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, 
                              dtype=torch.float16, device=input.device)
        v_output = torch.zeros(batch_size, kv_num_heads, seq_len, head_size, 
                              dtype=torch.float16, device=input.device)
        
        # 调用CUDA内核
        try:
            self._cuda_kernel.fused_ln_qkv_gptq_cuda(
                ctypes.c_void_p(input.data_ptr()),
                ctypes.c_void_p(qweight.data_ptr()),
                ctypes.c_void_p(qzeros.data_ptr()),
                ctypes.c_void_p(scales.data_ptr()),
                ctypes.c_void_p(ln_weight.data_ptr()),
                ctypes.c_void_p(ln_bias.data_ptr()),
                ctypes.c_void_p(q_output.data_ptr()),
                ctypes.c_void_p(k_output.data_ptr()),
                ctypes.c_void_p(v_output.data_ptr()),
                ctypes.c_int(batch_size),
                ctypes.c_int(seq_len),
                ctypes.c_int(hidden_dim),
                ctypes.c_int(num_heads),
                ctypes.c_int(kv_num_heads),
                ctypes.c_int(head_size),
                ctypes.c_int(groupsize),
                ctypes.c_float(eps)
            )
            
            logger.info(f"✅ 融合LN+QKV计算完成: {input.shape} -> QKV")
            return q_output, k_output, v_output
            
        except Exception as e:
            logger.error(f"❌ CUDA内核执行失败: {e}")
            raise RuntimeError(f"CUDA内核执行失败: {e}")
    
    def is_available(self) -> bool:
        """检查融合算子是否可用"""
        return self._cuda_kernel is not None

# 全局实例
_fused_ln_qkv_gptq = None

def get_fused_ln_qkv_gptq() -> FusedLNQKVGPTQ:
    """获取全局融合算子实例"""
    global _fused_ln_qkv_gptq
    if _fused_ln_qkv_gptq is None:
        _fused_ln_qkv_gptq = FusedLNQKVGPTQ()
    return _fused_ln_qkv_gptq

def fused_ln_qkv_gptq(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    num_heads: int,
    kv_num_heads: int,
    head_size: int,
    groupsize: int = 128,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    融合LayerNorm + QKV投影的便捷函数
    
    Args:
        input: 输入张量 [batch_size, seq_len, hidden_dim]
        qweight: GPTQ量化权重 [K//8, N]
        qzeros: GPTQ量化零点 [num_groups, groupsize//8]
        scales: GPTQ量化缩放 [num_groups, N]
        ln_weight: LayerNorm权重 [hidden_dim]
        ln_bias: LayerNorm偏置 [hidden_dim]
        num_heads: 注意力头数量
        kv_num_heads: KV头数量
        head_size: 头大小
        groupsize: GPTQ组大小
        eps: LayerNorm epsilon
        
    Returns:
        Tuple[Q, K, V]: QKV张量
    """
    fused_op = get_fused_ln_qkv_gptq()
    return fused_op.fused_ln_qkv_gptq(
        input, qweight, qzeros, scales, ln_weight, ln_bias,
        num_heads, kv_num_heads, head_size, groupsize, eps
    )
