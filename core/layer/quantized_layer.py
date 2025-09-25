import logging
import time
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention

# 设置日志记录
logger = logging.getLogger(__name__)


class Qwen7B4BitOptimizedLayerAdapter:
    """
    Qwen7B 4-bit量化模型的纯手动优化实现
    专注于最大化推理性能，减少反量化开销
    """

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

        # 模型维度
        self.hidden_size = getattr(self.config, "hidden_size", 4096)
        self.intermediate_size = getattr(self.config, "intermediate_size", 11008)

        # 性能计数器
        self.dequant_count = 0
        self.fused_ops = 0

        # 预分配优化缓冲区        self._setup_optimized_buffers()

        logger.info("🚀 初始化Qwen7B 4-bit量化模型纯手动优化层")

    def _setup_optimized_buffers(self):
        """为优化计算预分配CUDA缓冲区"""
        # 使用Pinned Memory提高传输速度
        pin_memory = True if self.device == "cuda:0" else False

        # 主要缓冲区
        self.buffer1 = torch.zeros(
            (1, 1, self.hidden_size * 3),
            device=self.device,
            dtype=torch.float16,
            pin_memory=pin_memory
        )
        self.buffer2 = torch.zeros(
            (1, 1, self.intermediate_size),
            device=self.device,
            dtype=torch.float16,
            pin_memory=pin_memory
        )
        self.buffer3 = torch.zeros(
            (1, 1, self.hidden_size),
            device=self.device,
            dtype=torch.float16,
            pin_memory=pin_memory
        )

        logger.debug("✅ 已分配优化CUDA缓冲区")

    def _fused_rms_norm(self, x, weight, eps=1e-6):
        """融合RMSNorm计算（无均值中心化）"""
        # 融合计算减少内核启动
        variance = ((x * x).mean(dim=-1, keepdim=True))
        x = weight * x / torch.sqrt(variance + eps)
        return x

    def _quantized_matmul(self, x, qweight, scales, qzeros, bias, out_features):
        """
        优化4-bit量化矩阵乘法
        融合反量化与矩阵乘法，减少内存访问
        """
        # 获取分组大小（GPTQ默认128）
        group_size = 128

        # 重塑量化权重 [out_features, in_features]
        orig_shape = qweight.shape
        in_features = orig_shape[1]

        # 计算分组数量
        num_groups = (orig_shape[0] + group_size - 1) // group_size

        # 调整scales和qzeros形状 [num_groups]
        if scales.shape[0] != num_groups:
            scales = scales.view(num_groups, -1)[:, 0]
        if qzeros.shape[0] != num_groups:
            qzeros = qzeros.view(num_groups, -1)[:, 0]

        # 输出张量 [batch_size, out_features]
        batch_size = x.shape[0]
        output = torch.zeros(
            (batch_size, out_features),
            device=x.device,
            dtype=torch.bfloat16
        )

        # 分块矩阵乘法（减少内存压力）
        chunk_size = min(1024, in_features)  # 优化块大小
        for i in range(0, in_features, chunk_size):
            end_i = min(i + chunk_size, in_features)
            x_chunk = x[:, i:end_i]  # [batch_size, chunk_size]

            # 计算当前块涉及的分组
            start_group = i // group_size
            end_group = (end_i + group_size - 1) // group_size

            for g in range(start_group, end_group):
                # 分组权重反量化
                start_row = g * group_size
                end_row = min(start_row + group_size, out_features)

                # 反量化当前分组 [group_size, chunk_size]
                group_weight = qweight[start_row:end_row, i:end_i]
                group_scale = scales[g]
                group_zero = qzeros[g]
                dequantized = (group_weight - group_zero) * group_scale

                # 矩阵乘法 [batch_size, group_size]
                output[:, start_row:end_row] += torch.matmul(
                    x_chunk,
                    dequantized.t()
                )

        # 添加偏置
        if bias is not None:
            output += bias.unsqueeze(0)

        self.dequant_count += 1
        return output

    def _optimized_mlp(self, hidden_states, mlp_norm_fn, mlp_fn):
        """优化MLP计算路径，融合反量化"""
        # 保存原始输入用于值计算
        original_hidden_states = hidden_states

        # 1. RMSNorm计算
        hidden_states = self._fused_rms_norm(
            hidden_states,
            mlp_norm_fn.weight
        )
        self.fused_ops += 1

        # 2. 门控路径（w1 -> act -> w2）
        gate = self._quantized_matmul(
            hidden_states,
            mlp_fn.w1.qweight,
            mlp_fn.w1.scales,
            mlp_fn.w1.qzeros,
            mlp_fn.w1.bias,
            self.intermediate_size
        )

        # 3. 激活函数（SiLU）
        gate = torch.nn.functional.silu(gate)

        # 4. 值路径（w3）
        value = self._quantized_matmul(
            original_hidden_states,
            mlp_fn.w3.qweight,
            mlp_fn.w3.scales,
            mlp_fn.w3.qzeros,
            mlp_fn.w3.bias,
            self.intermediate_size
        )

        # 5. 输出投影（w2）
        gate = self._quantized_matmul(
            gate,
            mlp_fn.w2.qweight,
            mlp_fn.w2.scales,
            mlp_fn.w2.qzeros,
            mlp_fn.w2.bias,
            self.hidden_size
        )

        # 6. 合并结果
        return gate * value

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
        hidden_states = self._fused_rms_norm(hidden_states, layer.ln_1.weight)
        norm_time = time.time() - start_time

        # 2. QKV计算（量化优化）
        qkv_start = time.time()

        # 使用融合矩阵乘法
        qkv = self._quantized_matmul(
            hidden_states,
            layer.attn.c_attn.qweight,
            layer.attn.c_attn.scales,
            layer.attn.c_attn.qzeros,
            layer.attn.c_attn.bias,
            self.hidden_size * 3
        )
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)

        qkv_time = time.time() - qkv_start

        # 3. 重塑形状
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

        # 5. 输出投影（量化优化）
        proj_start = time.time()
        attn_output = self._quantized_matmul(
            attn_output.reshape(batch_size, -1),
            layer.attn.c_proj.qweight,
            layer.attn.c_proj.scales,
            layer.attn.c_proj.qzeros,
            layer.attn.c_proj.bias,
            self.hidden_size
        ).unsqueeze(1)
        hidden_states = residual + attn_output
        proj_time = time.time() - proj_start

        # 6. MLP + 残差（量化优化）
        mlp_start = time.time()
        residual = hidden_states
        hidden_states = self._optimized_mlp(hidden_states, layer.ln_2, layer.mlp)
        hidden_states = residual + hidden_states
        mlp_time = time.time() - mlp_start

        # 记录性能
        total_time = time.time() - start_time
        if layer_idx == 0:
            logger.info(f"优化层 {layer_idx}: 总耗时 {total_time * 1000:.2f}ms | "
                        f"LN({norm_time * 1000:.2f}ms)+QKV({qkv_time * 1000:.2f}ms)+"
                        f"Reshape({reshape_time * 1000:.2f}ms)+Attn({attn_time * 1000:.2f}ms)+"
                        f"Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms)")
            logger.info(f"⚡ 量化指标 | 反量化: {self.dequant_count}次 | 融合操作: {self.fused_ops}")

        return hidden_states, (k.squeeze(2), v.squeeze(2))