
import logging
import time

import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
# 设置日志记录
logger = logging.getLogger(__name__)

class ModelLayerAdapter:


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
        hidden_states = norm_fn(hidden_states)

        norm_time = time.time() - norm_start
        logger.debug(f"Layer {layer_idx}: LayerNorm耗时 {norm_time * 1000:.2f}ms")

        # 3. QKV计算 (自动处理不同投影方式)
        qkv_start = time.time()


        # Qwen 7B: 合并的c_attn投影
        qkv = layer.attn.c_attn(hidden_states)
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)


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

        proj_fn = getattr(layer.self_attn if self.cfg["qkv_proj"] else layer.attn, self.cfg["proj"])
        attn_output = proj_fn(attn_output.reshape(batch_size, -1)).unsqueeze(1)  # [B, 1, D]
        hidden_states = residual + attn_output

        proj_time = time.time() - proj_start
        logger.debug(f"Layer {layer_idx}: 输出投影耗时 {proj_time * 1000:.2f}ms")

        # 7. MLP + 残差 (支持MoE)
        mlp_start = time.time()

        residual = hidden_states
        hidden_states = mlp_norm_fn(hidden_states)

        hidden_states = layer.mlp(hidden_states)  # 直接调用mlp模块
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

        return hidden_states, (k.squeeze(2), v.squeeze(2))  # [B, H, D]