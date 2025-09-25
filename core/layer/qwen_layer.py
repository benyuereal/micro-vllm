import logging
import time
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention

logger = logging.getLogger(__name__)


class QwenModelLayerAdapter:
    # 仅支持Qwen模型配置
    QWEN_CONFIG = {
        "norm": "ln_1", "attn": "c_attn", "proj": "c_proj", "mlp_norm": "ln_2",
        "qkv_split": True, "qkv_proj": False,
        "mlp": "mlp", "residual": True,
    }

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int,
                 is_gptq_model: bool = True):
        self.config = model_config
        self.device = device
        self.model_type = model_config.model_type
        self.num_heads, self.head_size, self.kv_num_heads = num_heads, head_size, kv_num_heads
        self.is_gptq_model = is_gptq_model  # GPTQ优化标志

        # 优化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

        self.cfg = self.QWEN_CONFIG

    def process_layer(self,
                      layer,
                      hidden_states: torch.Tensor,
                      cache_manager,
                      seq_ids: List[int],
                      context_lens: List[int],
                      token_positions: Optional[torch.Tensor] = None,
                      layer_idx: int = 0,
                      current_positions: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # 对于GPTQ模型，减少性能分析开销
        start_time = time.time()

        # 1. LayerNorm - 使用inplace操作
        residual = hidden_states
        norm_fn = getattr(layer, self.cfg["norm"])
        hidden_states = norm_fn(hidden_states)

        # 2. QKV计算 - Qwen特定优化
        qkv_start = time.time()
        qkv = layer.attn.c_attn(hidden_states)
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)
        qkv_time = time.time() - qkv_start

        # 3. 重塑形状 - 使用view避免拷贝
        batch_size, seq_len, _ = hidden_states.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).transpose(1, 2)

        # 4. 注意力计算
        attn_start = time.time()
        attn_output = self.attention(
            query=q, cache_manager=cache_manager, seq_ids=seq_ids,
            context_lens=context_lens, layer_idx=layer_idx, key=k, value=v
        )
        attn_time = time.time() - attn_start

        # 5. 输出投影 + 残差
        proj_start = time.time()
        proj_fn = getattr(layer.attn, self.cfg["proj"])
        attn_output = proj_fn(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        hidden_states = residual + attn_output
        proj_time = time.time() - proj_start

        # 6. MLP + 残差
        mlp_start = time.time()
        residual = hidden_states
        mlp_norm_fn = getattr(layer, self.cfg["mlp_norm"])
        hidden_states = mlp_norm_fn(hidden_states)

        mlp_fn = getattr(layer, self.cfg["mlp"])
        hidden_states = mlp_fn(hidden_states)
        hidden_states = residual + hidden_states
        mlp_time = time.time() - mlp_start

        # 记录耗时（仅非GPTQ模型）
        if layer_idx == 0:
            total_time = time.time() - start_time
            logger.info(f"Layer {layer_idx}: 总处理耗时 {total_time * 1000:.2f}ms, "
                        f"分布: LN({(time.time() - start_time - qkv_time - attn_time - proj_time - mlp_time) * 1000:.2f}ms)+"
                        f"QKV({qkv_time * 1000:.2f}ms)+Attn({attn_time * 1000:.2f}ms)+"
                        f"Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms)")

        return hidden_states, (k, v)