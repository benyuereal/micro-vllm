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

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int = None):
        """
        初始化 Qwen7B 量化适配器

        参数:
            model_config: 模型配置
            device: 设备 ("cuda", "mps", "cpu")
            num_heads: 注意力头数
            head_size: 每个头维度
            kv_num_heads: KV头数 (GQA支持)，Qwen7B默认为None
        """
        self.config = model_config
        self.device = device
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads if kv_num_heads is not None else num_heads

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=self.kv_num_heads,
            device=device
        )

        # 量化相关初始化
        self.quant_group_size = self._get_quant_group_size()
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
            batch_size, seq_len, _ = hidden_states.shape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)

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
            # 重塑为 [B, S, D]
            batch_size, seq_len, _ = hidden_states.shape
            attn_output = attn_output.view(batch_size, seq_len, -1)
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
                f"Attn({attn_time * 1000:.2f}ms)+Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms) | "
                f"Quant: {self._is_quantized}"
            )

        return hidden_states, (k, v)

    def _get_quant_group_size(self) -> int:
        """获取量化组大小"""
        # 从模型配置获取
        if hasattr(self.config, "quantization_config"):
            quant_config = self.config.quantization_config
            if isinstance(quant_config, dict) and "group_size" in quant_config:
                return quant_config["group_size"]

        # 从配置属性获取
        if hasattr(self.config, "group_size"):
            return self.config.group_size

        # 默认值
        return 128

    def _detect_qwen_quantized_model(self, layer) -> bool:
        """
        检测是否为 Qwen7B INT4 量化模型
        """
        # 检查是否为 Qwen7B 结构
        if not hasattr(layer, "attn") or not hasattr(layer.attn, "c_attn"):
            return False

        # 检查是否为量化权重
        c_attn = layer.attn.c_attn
        logger.info("c_attn is ", c_attn)
        # 情况1: GPTQ-for-LLaMa 格式 (有 qweight 属性)
        if hasattr(c_attn, "qweight"):
            return True

        # 情况2: 标准 GPTQ 格式 (权重类型为 int8)
        if hasattr(c_attn, "weight") and c_attn.weight.dtype == torch.int8:
            return True

            # 情况3: auto_gptq 的 QuantLinear        if hasattr(c_attn, "bits") and c_attn.bits == 4:
            return True

        return False

    def _quantized_qkv_proj(self, layer, hidden_states: torch.Tensor):
        """
        Qwen7B INT4 量化 QKV 投影
        """
        c_attn = layer.attn.c_attn

        # 安全获取量化参数
        qkv_weight, qkv_scale, qkv_zero = self._get_quant_params(c_attn)

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
        c_proj = layer.attn.c_proj

        # 安全获取量化参数
        out_weight, out_scale, out_zero = self._get_quant_params(c_proj)

        # 调用融合内核
        return QuantKernels.fused_quant_out_proj(
            attn_output=attn_output,
            out_weight=out_weight,
            out_scale=out_scale,
            out_zero=out_zero,
            group_size=self.quant_group_size
        )

    def _get_quant_params(self, module):
        """
        安全获取量化参数 (兼容多种量化格式)
        """
        # 1. GPTQ-for-LLaMa 格式
        if hasattr(module, "qweight"):
            weight = module.qweight
            scale = module.scales
            zero = module.qzeros
            return weight, scale, zero

        # 2. auto_gptq 的 QuantLinear
        if hasattr(module, "weight") and hasattr(module, "scales"):
            weight = module.weight
            scale = module.scales
            zero = getattr(module, "zeros", None) or getattr(module, "qzeros", None)
            if zero is None:
                # 计算默认零点 (假设对称量化)
                zero = torch.zeros_like(scale)
            return weight, scale, zero

        # 3. 标准 GPTQ 格式
        if hasattr(module, "weight") and hasattr(module, "scale"):
            weight = module.weight
            scale = module.scale
            zero = getattr(module, "zero", None)
            if zero is None:
                # 计算默认零点
                zero = torch.zeros_like(scale)
            return weight, scale, zero

        # 4. 回退方案: 尝试常见属性名
        weight = getattr(module, "weight", None)
        scale = getattr(module, "scale", getattr(module, "scales", None))
        zero = getattr(module, "zero", getattr(module, "zeros", getattr(module, "qzeros", None)))

        if weight is not None and scale is not None and zero is not None:
            return weight, scale, zero

        # 5. 最后尝试: 检查缓冲区
        if hasattr(module, "_buffers"):
            buffers = module._buffers
            if "qweight" in buffers:
                return buffers["qweight"], buffers["scales"], buffers["qzeros"]
            elif "weight" in buffers and "scales" in buffers:
                return buffers["weight"], buffers["scales"], buffers.get("zeros", buffers.get("qzeros"))

        raise AttributeError(f"无法获取量化参数: {module.__class__.__name__}")

    def _safe_reshape_attn_output(self, attn_output: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        安全重塑注意力输出形状
        """
        try:
            # 尝试标准重塑
            return attn_output.view(batch_size, seq_len, -1)
        except Exception as e:
            # 如果失败，尝试推断隐藏维度
            hidden_dim = attn_output.shape[-1] * attn_output.shape[-2]
            return attn_output.view(batch_size, seq_len, hidden_dim)