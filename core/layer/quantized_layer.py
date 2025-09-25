import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention


class QwenLayerAdapter:
    # 精简模型配置
    MODEL_CONFIGS = {
        "qwen": {
            "norm": "ln_1", "proj": "c_proj",
            "mlp_norm": "ln_2", "qkv_split": True
        },
        "qwen2": {
            "norm": "input_layernorm", "proj": "o_proj",
            "mlp_norm": "post_attention_layernorm", "qkv_split": False
        }
    }

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int):
        self.config = model_config
        self.device = device
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.model_type = model_config.model_type

        # 初始化PagedAttention
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
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
        """INT4优化后的层处理函数"""
        # 1. LayerNorm + 残差
        residual = hidden_states
        hidden_states = getattr(layer, self.cfg["norm"])(hidden_states)

        # 2. QKV计算 (INT4-aware)
        qkv = layer.attn.c_attn(hidden_states)  # 自动使用INT4优化
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)

        # 3. 重塑形状 + 内存布局优化
        batch_size, seq_len, _ = hidden_states.shape

        # [B, S, H, D] -> [B, H, S, D] 并优化内存布局
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)

        # INT4内存布局优化
        q = q.to(memory_format=torch.channels_last)
        k = k.to(memory_format=torch.channels_last)
        v = v.to(memory_format=torch.channels_last)

        # 4. 注意力计算 (使用现有PagedAttention)
        attn_output = self.attention(
            query=q.squeeze(2),  # [B, H, D]
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=k.squeeze(2),  # [B, H, D]
            value=v.squeeze(2)  # [B, H, D]
        )

        # 5. 输出投影 (INT4-aware)
        proj = getattr(layer.attn, self.cfg["proj"])
        attn_output = proj(attn_output.reshape(batch_size, -1)).unsqueeze(1)  # [B, 1, D]
        hidden_states = residual + attn_output

        # 6. MLP (INT4-aware)
        residual = hidden_states
        hidden_states = getattr(layer, self.cfg["mlp_norm"])(hidden_states)
        hidden_states = layer.mlp(hidden_states)  # 自动使用INT4优化
        hidden_states = residual + hidden_states

        return hidden_states, (k.squeeze(2), v.squeeze(2))  # [B, H, D]