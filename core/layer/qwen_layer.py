# File: QwenModelLayerAdapter.py
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
import logging
import time

# 导入量化内核
from .quant_kernels import QuantKernels

# 设置日志记录
logger = logging.getLogger(__name__)


class QwenModelLayerAdapter:
    """
    Qwen7B INT4 量化模型专用适配器
    专注消除量化模型的反量化开销
    """

    def __init__(self, model_config, device: str, num_heads: int, head_size: int):
        """
        初始化 Qwen7B 量化适配器

        参数:
            model_config: 模型配置
            device: 设备 ("cuda", "mps", "cpu")
            num_heads: 注意力头数
            head_size: 每个头维度
        """
        self.config = model_config
        self.device = device
        self.num_heads = num_heads
        self.head_size = head_size

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=num_heads,  # Qwen7B 无 GQA
            device=device
        )

        # 量化相关初始化
        self.quant_group_size = getattr(model_config, "group_size", 128)
        self._is_quantized = False

    def process_layer(self,
                      layer,
                      hidden_states: torch.Tensor,  # [B, S, D]
                      cache_manager,
                      seq_ids: List[int],
                      context_lens: List[int],
                      token_positions: Optional[torch.Tensor] = None,
                      layer_idx: int = 0,
                      current_positions: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # 记录开始时间
        start_time = time.time()

        # 检测是否为量化模型
        self._is_quantized = self._detect_qwen_quantized_model(layer)

        # 1. LayerNorm + 残差
        residual = hidden_states
        hidden_states = layer.ln_1(hidden_states)
        norm_time = time.time() - start_time

        # 2. QKV 计算 (量化优化)
        qkv_start = time.time()

        if self._is_quantized:
            q, k, v = self._quantized_qkv_proj(layer, hidden_states)
        else:
            # 标准 QKV 计算
            qkv = layer.attn.c_attn(hidden_states)
            hidden_size = qkv.shape[-1] // 3
            q, k, v = qkv.split(hidden_size, dim=-1)
            q = q.view(-1, self.num_heads, self.head_size).transpose(0, 1)
            k = k.view(-1, self.num_heads, self.head_size).transpose(0, 1)
            v = v.view(-1, self.num_heads, self.head_size).transpose(0, 1)

        qkv_time = time.time() - qkv_start

        # 3. 注意力计算
        attn_start = time.time()
        attn_output = self.attention(
            query=q,
            key=k,
            value=v,
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx
        )
        attn_time = time.time() - attn_start

        # 4. 输出投影 (量化优化)
        proj_start = time.time()

        if self._is_quantized:
            attn_output = self._quantized_out_proj(layer, attn_output)
        else:
            attn_output = layer.attn.c_proj(attn_output)

        proj_time = time.time() - proj_start

        # 5. 残差连接
        hidden_states = residual + attn_output

        # 6. MLP
        mlp_start = time.time()
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        mlp_time = time.time() - mlp_start

        # 记录耗时
        if layer_idx == 0:
            total_time = time.time() - start_time
            logger.info(
                f"Layer {layer_idx}: 总耗时 {total_time * 1000:.2f}ms | "
                f"LN1({norm_time * 1000:.2f}ms)+QKV({qkv_time * 1000:.2f}ms)+"
                f"Attn({attn_time * 1000:.2f}ms)+Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms)"
            )

        return hidden_states, (k, v)

    def _detect_qwen_quantized_model(self, layer) -> bool:
        """
        检测是否为 Qwen7B INT4 量化模型
        """
        # 检查是否为 Qwen7B 结构
        if not hasattr(layer, "attn") or not hasattr(layer.attn, "c_attn"):
            return False

        # 检查是否为量化权重
        weight = layer.attn.c_attn.weight
        return hasattr(weight, "qweight") or weight.dtype == torch.int8

    def _quantized_qkv_proj(self, layer, hidden_states: torch.Tensor):
        """
        Qwen7B INT4 量化 QKV 投影
        """
        # 获取量化参数
        weight = layer.attn.c_attn.weight

        if hasattr(weight, "qweight"):
            # GPTQ-for-LLaMa 格式
            qkv_weight = weight.qweight
            qkv_scale = weight.scales
            qkv_zero = weight.qzeros
        else:
            # 标准 GPTQ 格式
            qkv_weight = weight
            qkv_scale = getattr(layer.attn.c_attn, "scale", None)
            qkv_zero = getattr(layer.attn.c_attn, "zero", None)

        # 调用融合内核
        return QuantKernels.fused_quant_qkv_proj(
            hidden_states=hidden_states,
            qkv_weight=qkv_weight,
            qkv_scale=qkv_scale,
            qkv_zero=qkv_zero,
            num_heads=self.num_heads,
            head_dim=self.head_size,
            group_size=self.quant_group_size
        )

    def _quantized_out_proj(self, layer, attn_output: torch.Tensor):
        """
        Qwen7B INT4 量化输出投影
        """
        # 获取量化参数
        weight = layer.attn.c_proj.weight

        if hasattr(weight, "qweight"):
            # GPTQ-for-LLaMa 格式
            out_weight = weight.qweight
            out_scale = weight.scales
            out_zero = weight.qzeros
        else:
            # 标准 GPTQ 格式
            out_weight = weight
            out_scale = getattr(layer.attn.c_proj, "scale", None)
            out_zero = getattr(layer.attn.c_proj, "zero", None)

        # 调用融合内核
        return QuantKernels.fused_quant_out_proj(
            attn_output=attn_output,
            out_weight=out_weight,
            out_scale=out_scale,
            out_zero=out_zero,
            group_size=self.quant_group_size
        )