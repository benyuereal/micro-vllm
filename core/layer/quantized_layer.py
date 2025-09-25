import logging
import time
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
import numpy as np

# 设置日志记录
logger = logging.getLogger(__name__)


class QuantizedModelLayerAdapter:
    # 模型架构配置 (可扩展)
    MODEL_CONFIGS = {
        "qwen": {  # Qwen 7B
            "norm": "ln_1", "attn": "c_attn", "proj": "c_proj", "mlp_norm": "ln_2",
            "qkv_split": True, "qkv_proj": False,
            "mlp": "mlp", "residual": True,
        },
        "qwen2": {  # Qwen 1.5/2.5
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
        },
        "qwen3": {  # Qwen3 (与Qwen2相同，但支持MoE)
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
            "moe": True,  # ✅ 支持MoE
        },
    }

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int):
        self.config = model_config
        self.device = device
        self.model_type = model_config.model_type
        self.num_heads, self.head_size, self.kv_num_heads = num_heads, head_size, kv_num_heads

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

        # 验证模型类型
        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.cfg = self.MODEL_CONFIGS[self.model_type]

        # 量化模型优化配置
        self.is_quantized = hasattr(model_config, "quantization_config")
        print("Quantization Config:", self.is_quantized)
        logger.info(f"model Quantization config: {model_config}")
        self.quant_bits = 4
        logger.info(f"检测到量化模型: {self.quant_bits}-bit量化")
        # 预分配优化缓冲区
        self._setup_optimization_buffers()

        # 性能计数器
        self.dequant_count = 0
        self.fused_ops = 0

    def _setup_optimization_buffers(self):
        """为量化模型预分配优化缓冲区"""
        if not self.is_quantized or self.quant_bits != 4:
            return

        # 预分配反量化缓冲区
        hidden_size = getattr(self.config, "hidden_size", 4096)
        intermediate_size = getattr(self.config, "intermediate_size", 11008)

        # MLP反量化缓冲区
        self.mlp_dequant_buffer = torch.zeros(
            1, 1, intermediate_size,
            device=self.device,
            dtype=torch.bfloat16
        )

        # QKV反量化缓冲区
        self.qkv_dequant_buffer = torch.zeros(
            1, 1, hidden_size * 3,
            device=self.device,
            dtype=torch.bfloat16
        )

        logger.debug("✅ 已分配量化优化缓冲区")

    def _quantized_layer_norm(self, x, weight, bias, eps):
        """针对量化模型的优化LayerNorm"""
        # 使用量化友好的计算方式
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # 量化模型使用更稳定的计算
        std = (var + eps).sqrt()
        x = (x - mean) / std

        # 应用缩放和偏移
        return weight * x + bias

    def _fused_qkv_projection(self, hidden_states, layer):
        """融合QKV投影优化"""
        # 针对GPTQ量化模型的优化
        if (self.is_quantized and self.quant_bits == 4 and
                hasattr(layer.attn, "c_attn") and
                hasattr(layer.attn.c_attn, "qweight")):

            # 直接使用量化权重计算
            qkv = self._quantized_linear(
                hidden_states,
                layer.attn.c_attn,
                self.qkv_dequant_buffer
            )
            hidden_size = qkv.shape[-1] // 3
            self.dequant_count += 1
            self.fused_ops += 1
            return qkv.split(hidden_size, dim=-1)
        else:
            # 标准计算
            qkv = layer.attn.c_attn(hidden_states)
            hidden_size = qkv.shape[-1] // 3
            return qkv.split(hidden_size, dim=-1)

    def _quantized_linear(self, x, quantized_linear_layer, buffer):
        """优化量化线性层计算"""
        # 获取量化参数
        qweight = quantized_linear_layer.qweight
        scales = quantized_linear_layer.scales
        qzeros = quantized_linear_layer.qzeros

        # 使用预分配缓冲区避免重复分配
        if buffer.shape != qweight.shape:
            buffer = torch.empty_like(qweight, dtype=torch.float16)

        # 基于位深度的反量化
        if self.quant_bits == 4:
            # 4-bit反量化优化
            group_size = scales.shape[0]
            orig_shape = qweight.shape
            qweight = qweight.reshape(group_size, -1)
            scales = scales.view(-1, 1)
            qzeros = qzeros.view(-1, 1)

            # 高效反量化计算
            dequantized = (qweight - qzeros) * scales
            buffer = dequantized.reshape(orig_shape).to(torch.bfloat16)
        else:
            # 其他位深度使用标准反量化
            buffer = quantized_linear_layer.dequantize()

        # 融合计算
        return torch.nn.functional.linear(x, buffer, quantized_linear_layer.bias)

    def _optimized_mlp(self, hidden_states, mlp_norm_fn, mlp_fn):
        """优化后的MLP计算路径 - 修复版本"""
        # 保存原始输入（用于某些模型结构）
        original_hidden_states = hidden_states

        # 1. 融合LayerNorm计算
        if hasattr(mlp_norm_fn, "weight"):
            # 手动融合LayerNorm计算
            mean = hidden_states.mean(dim=-1, keepdim=True)
            variance = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
            std = (variance + mlp_norm_fn.variance_epsilon).sqrt()
            hidden_states = mlp_norm_fn.weight * (hidden_states - mean) / std + mlp_norm_fn.bias

        # 2. 第一层线性变换（门控）
        if (self.is_quantized and self.quant_bits == 4 and
                hasattr(mlp_fn, "w1") and hasattr(mlp_fn.w1, "qweight")):
            # 使用量化优化计算
            gate = self._quantized_linear(
                hidden_states,
                mlp_fn.w1,
                self.mlp_dequant_buffer
            )
            self.dequant_count += 1
        else:
            gate = mlp_fn.w1(hidden_states)

        # 3. 激活函数
        if hasattr(mlp_fn, "act_fn"):
            gate = mlp_fn.act_fn(gate)
        else:
            # 默认GELU激活
            gate = torch.nn.functional.gelu(gate)

        # 4. 第二层线性变换（值）
        if (self.is_quantized and self.quant_bits == 4 and
                hasattr(mlp_fn, "w3") and hasattr(mlp_fn.w3, "qweight")):
            # 注意：某些模型使用原始输入计算值
            value_input = original_hidden_states if hasattr(mlp_fn, "w3") else hidden_states
            value = self._quantized_linear(
                value_input,
                mlp_fn.w3,
                self.mlp_dequant_buffer
            )
            self.dequant_count += 1
        else:
            value = mlp_fn.w3(original_hidden_states)

        # 5. 第三层线性变换（输出）
        if (self.is_quantized and self.quant_bits == 4 and
                hasattr(mlp_fn, "w2") and hasattr(mlp_fn.w2, "qweight")):
            gate = self._quantized_linear(
                gate,
                mlp_fn.w2,
                self.mlp_dequant_buffer
            )
            self.dequant_count += 1
        else:
            gate = mlp_fn.w2(gate)

        # 6. 合并结果
        hidden_states = gate * value

        return hidden_states

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

        # 1. 自动适配模型架构
        norm_fn = getattr(layer, self.cfg["norm"])
        mlp_norm_fn = getattr(layer, self.cfg["mlp_norm"])
        mlp_fn = getattr(layer, self.cfg["mlp"])

        # 记录LayerNorm前的时间
        norm_start = time.time()

        # 2. LayerNorm + 残差
        residual = hidden_states

        # 量化模型使用优化版LayerNorm
        if self.is_quantized and self.quant_bits == 4:
            hidden_states = self._quantized_layer_norm(
                hidden_states,
                norm_fn.weight,
                norm_fn.bias,
                norm_fn.variance_epsilon
            )
            self.fused_ops += 1
        else:
            hidden_states = norm_fn(hidden_states)

        norm_time = time.time() - norm_start
        logger.debug(f"Layer {layer_idx}: LayerNorm耗时 {norm_time * 1000:.2f}ms")

        # 3. QKV计算 (自动处理不同投影方式)
        qkv_start = time.time()

        # 使用优化QKV投影
        q, k, v = self._fused_qkv_projection(hidden_states, layer)

        qkv_time = time.time() - qkv_start
        logger.debug(f"Layer {layer_idx}: QKV投影耗时 {qkv_time * 1000:.2f}ms")

        # 4. 重塑形状 [B, S, D] → [B, H, D]
        reshape_start = time.time()

        batch_size, seq_len, _ = hidden_states.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)

        reshape_time = time.time() - reshape_start
        logger.debug(f"Layer {layer_idx}: 形状重塑耗时 {reshape_time * 1000:.2f}ms")

        # 5. 注意力计算 (零拷贝)
        attn_start = time.time()

        attn_output = self.attention(
            query=q.squeeze(2),  # [B, H, D]
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=k.squeeze(2),  # [B, H, D]
            value=v.squeeze(2)  # [B, H, D]
        )

        attn_time = time.time() - attn_start
        logger.debug(f"Layer {layer_idx}: 注意力计算耗时 {attn_time * 1000:.2f}ms")

        # 6. 输出投影 + 残差
        proj_start = time.time()

        # 确定投影函数
        if self.cfg["qkv_proj"]:
            proj_fn = getattr(layer.self_attn, self.cfg["proj"])
        else:
            proj_fn = getattr(layer.attn, self.cfg["proj"])

        # 优化量化模型的投影计算
        if (self.is_quantized and self.quant_bits == 4 and
                hasattr(proj_fn, "qweight")):

            attn_output = self._quantized_linear(
                attn_output.reshape(batch_size, -1),
                proj_fn,
                self.mlp_dequant_buffer
            ).unsqueeze(1)
            self.dequant_count += 1
        else:
            attn_output = proj_fn(attn_output.reshape(batch_size, -1)).unsqueeze(1)  # [B, 1, D]

        hidden_states = residual + attn_output

        proj_time = time.time() - proj_start
        logger.debug(f"Layer {layer_idx}: 输出投影耗时 {proj_time * 1000:.2f}ms")

        # 7. MLP + 残差 (支持MoE)
        mlp_start = time.time()

        residual = hidden_states
        hidden_states = mlp_norm_fn(hidden_states)

        # 使用优化路径处理量化模型的MLP
        if self.is_quantized and self.quant_bits == 4:
            hidden_states = self._optimized_mlp(hidden_states, mlp_norm_fn, mlp_fn)
        else:
            hidden_states = mlp_fn(hidden_states)  # 原始实现

        hidden_states = residual + hidden_states

        mlp_time = time.time() - mlp_start
        logger.debug(f"Layer {layer_idx}: MLP计算耗时 {mlp_time * 1000:.2f}ms")

        # 记录总耗时
        total_time = time.time() - start_time
        if layer_idx == 0:
            logger.info(f"Layer {layer_idx}: 总处理耗时 {total_time * 1000:.2f}ms, "
                        f"分布: LN({norm_time * 1000:.2f}ms)+QKV({qkv_time * 1000:.2f}ms)+"
                        f"Reshape({reshape_time * 1000:.2f}ms)+Attn({attn_time * 1000:.2f}ms)+"
                        f"Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms)")

            # 记录量化特定指标
            if self.is_quantized and self.quant_bits == 4:
                logger.info(f"量化模型指标 | 反量化次数: {self.dequant_count} | 融合操作: {self.fused_ops}")

        return hidden_states, (k.squeeze(2), v.squeeze(2))  # [B, H, D]