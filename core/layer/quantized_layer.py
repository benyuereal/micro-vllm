import logging
import time
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention

# 设置日志记录
logger = logging.getLogger(__name__)


class Qwen7B4BitLayerAdapter:
    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int):
        self.config = model_config
        self.device = device
        self.num_heads, self.head_size, self.kv_num_heads = num_heads, head_size, kv_num_heads

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

        # 预分配优化缓冲区（专为4-bit量化设计）
        self._setup_quantization_buffers()

        # 性能计数器
        self.dequant_count = 0
        self.fused_ops = 0
        logger.info("✅ 初始化Qwen7B 4-bit量化模型优化层")

    def _setup_quantization_buffers(self):
        """为4-bit量化模型预分配优化缓冲区"""
        # 获取模型维度
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

        # 输出投影反量化缓冲区
        self.proj_dequant_buffer = torch.zeros(
            1, 1, hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )

        logger.debug("✅ 已分配4-bit量化优化缓冲区")

    def _quantized_linear(self, x, quantized_linear_layer, buffer):
        """优化4-bit量化线性层计算 - 修复维度问题"""
        # 获取量化参数
        qweight = quantized_linear_layer.qweight
        scales = quantized_linear_layer.scales
        qzeros = quantized_linear_layer.qzeros

        # 使用预分配缓冲区避免重复分配
        if buffer.shape != qweight.shape:
            buffer = torch.empty_like(qweight, dtype=torch.float16)

        # 获取分组大小（从量化层获取）
        group_size = getattr(quantized_linear_layer, "group_size", 128)

        # 4-bit反量化优化
        orig_shape = qweight.shape

        # 计算分组数量
        num_groups = (orig_shape[0] + group_size - 1) // group_size

        # 确保scales和qzeros有正确的形状
        if scales.shape[0] != num_groups:
            # 调整scales形状
            scales = scales.view(num_groups, -1)[:, 0]
        if qzeros.shape[0] != num_groups:
            # 调整qzeros形状
            qzeros = qzeros.view(num_groups, -1)[:, 0]

        # 重塑权重为分组格式
        padded_weight = torch.zeros(num_groups * group_size, orig_shape[1],
                                    device=qweight.device, dtype=qweight.dtype)
        padded_weight[:orig_shape[0], :] = qweight

        # 执行反量化
        dequantized = torch.zeros_like(padded_weight, dtype=torch.float16)
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, orig_shape[0])

            if end_idx > start_idx:
                group_weight = padded_weight[start_idx:end_idx, :]
                group_scale = scales[i]
                group_zero = qzeros[i]

                # 反量化计算
                dequantized_group = (group_weight - group_zero) * group_scale
                dequantized[start_idx:end_idx, :] = dequantized_group

        # 复制到缓冲区
        buffer = dequantized[:orig_shape[0], :].to(torch.bfloat16)

        self.dequant_count += 1
        return torch.nn.functional.linear(x, buffer, quantized_linear_layer.bias)

    def _optimized_mlp(self, hidden_states, mlp_norm_fn, mlp_fn):
        """Qwen7B 4-bit量化模型优化的MLP计算路径"""
        # 保存原始输入（用于值计算）
        original_hidden_states = hidden_states

        # 1. RMSNorm计算（无均值中心化）
        variance = ((hidden_states * hidden_states).mean(dim=-1, keepdim=True))
        hidden_states = mlp_norm_fn.weight * hidden_states / torch.sqrt(variance + 1e-6)

        self.fused_ops += 1

        # 2. 门控线性变换 (w1)
        gate = self._quantized_linear(
            hidden_states,
            mlp_fn.w1,
            self.mlp_dequant_buffer
        )

        # 3. 值线性变换 (w3) - 使用原始输入
        value = self._quantized_linear(
            original_hidden_states,
            mlp_fn.w3,
            self.mlp_dequant_buffer
        )

        # 4. 激活函数 (SiLU)
        gate = torch.nn.functional.silu(gate)

        # 5. 输出线性变换 (w2)
        gate = self._quantized_linear(
            gate,
            mlp_fn.w2,
            self.mlp_dequant_buffer
        )

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

        # 1. RMSNorm + 残差
        residual = hidden_states

        # 优化版RMSNorm（无均值中心化）
        variance = ((hidden_states * hidden_states).mean(dim=-1, keepdim=True))
        hidden_states = layer.ln_1.weight * hidden_states / torch.sqrt(variance + 1e-6)

        norm_time = time.time() - start_time
        self.fused_ops += 1

        # 2. QKV计算 (量化优化)
        qkv_start = time.time()

        # 使用量化优化计算
        qkv = self._quantized_linear(
            hidden_states,
            layer.attn.c_attn,
            self.qkv_dequant_buffer
        )
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)

        qkv_time = time.time() - qkv_start

        # 3. 重塑形状 [B, S, D] → [B, H, D]
        reshape_start = time.time()

        batch_size, seq_len, _ = hidden_states.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)

        reshape_time = time.time() - reshape_start

        # 4. 注意力计算
        attn_start = time.time()

        attn_output = self.attention(
            query=q.squeeze(2),
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=k.squeeze(2),
            value=v.squeeze(2)
        )

        attn_time = time.time() - attn_start

        # 5. 输出投影 + 残差 (量化优化)
        proj_start = time.time()

        attn_output = self._quantized_linear(
            attn_output.reshape(batch_size, -1),
            layer.attn.c_proj,
            self.proj_dequant_buffer
        ).unsqueeze(1)

        hidden_states = residual + attn_output

        proj_time = time.time() - proj_start

        # 6. MLP + 残差 (量化优化)
        mlp_start = time.time()

        residual = hidden_states

        # 优化版RMSNorm
        variance = ((hidden_states * hidden_states).mean(dim=-1, keepdim=True))
        hidden_states = layer.ln_2.weight * hidden_states / torch.sqrt(variance + 1e-6)

        # 优化MLP计算
        hidden_states = self._optimized_mlp(hidden_states, layer.ln_2, layer.mlp)

        hidden_states = residual + hidden_states

        mlp_time = time.time() - mlp_start

        # 记录总耗时
        total_time = time.time() - start_time
        if layer_idx == 0:
            logger.info(f"Layer {layer_idx}: 总处理耗时 {total_time * 1000:.2f}ms, "
                        f"分布: LN({norm_time * 1000:.2f}ms)+QKV({qkv_time * 1000:.2f}ms)+"
                        f"Reshape({reshape_time * 1000:.2f}ms)+Attn({attn_time * 1000:.2f}ms)+"
                        f"Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms)")

            # 记录量化指标
            logger.info(f"4-bit量化指标 | 反量化次数: {self.dequant_count} | 融合操作: {self.fused_ops}")

        return hidden_states, (k.squeeze(2), v.squeeze(2))