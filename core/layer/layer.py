import logging
from typing import Optional

from core.kv_store import KVStore
from core.paged_attention import PagedAttention
# 导入您现有的模块
from models.qwen_adapter import QwenModelAdapter



class ModelLayerAdapter:
    """模型层适配器，处理不同架构的模型层计算"""

    def __init__(self, model_config, device, num_heads: int, head_size: int, kv_num_heads: int, kv_store: Optional[KVStore] = None):
        self.model_config = model_config
        self.device = device
        self.model_type = model_config.model_type
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

    def process_layer(self, layer, hidden_states, cache_manager, seq_ids,
                      context_lens, token_positions, layer_idx, current_positions):
        """处理单层计算，返回更新后的hidden_states和当前层的KV"""

        if self.model_type == "qwen":  # Qwen 7B架构
            return self._process_qwen_layer(layer, hidden_states, cache_manager, seq_ids,
                                            context_lens, token_positions, layer_idx, current_positions)
        elif self.model_type == "qwen2":  # Qwen 1.5 0.5B架构
            return self._process_qwen2_layer(layer, hidden_states, cache_manager, seq_ids,
                                             context_lens, token_positions, layer_idx, current_positions)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _process_qwen_layer(self, layer, hidden_states, cache_manager, seq_ids,
                            context_lens, token_positions, layer_idx, current_positions):
        """处理Qwen 7B架构的层计算"""
        # Qwen 7B使用transformer.h和不同的层结构
        residual = hidden_states
        hidden_states = layer.ln_1(hidden_states)

        # Qwen 7B使用合并的c_attn投影
        qkv = layer.attn.c_attn(hidden_states)
        q, k, v = qkv.split(self.model_config.hidden_size, dim=2)

        # 重塑形状
        batch_size = hidden_states.size(0)
        num_heads = self.num_heads
        head_size = self.head_size

        query = q.view(batch_size, num_heads, head_size)
        key = k.view(batch_size, num_heads, head_size)
        value = v.view(batch_size, num_heads, head_size)

        # 保存当前层的KV
        current_k = key
        current_v = value

        # 注意力计算
        attn_output = self.attention.forward(
            query=query,
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=current_k,
            value=current_v
        )

        attn_output = attn_output.reshape(attn_output.size(0), -1)
        attn_output = layer.attn.c_proj(attn_output)
        attn_output = attn_output.unsqueeze(1)
        hidden_states = residual + attn_output

        # 前馈网络
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, (current_k, current_v)

    def _process_qwen2_layer(self, layer, hidden_states, cache_manager, seq_ids,
                             context_lens, token_positions, layer_idx, current_positions):
        """处理Qwen 1.5 0.5B架构的层计算"""
        # Qwen 1.5使用model.layers和不同的层结构
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # Qwen 1.5使用分开的q_proj, k_proj, v_proj
        query = layer.self_attn.q_proj(hidden_states)
        key = layer.self_attn.k_proj(hidden_states)
        value = layer.self_attn.v_proj(hidden_states)

        # 重塑形状
        batch_size = hidden_states.size(0)


        query = query.view(batch_size, self.num_heads, self.head_size)
        key = key.view(batch_size, self.kv_num_heads, self.head_size)
        value = value.view(batch_size, self.kv_num_heads, self.head_size)

        # 保存当前层的KV
        current_k = key
        current_v = value

        # 注意力计算
        attn_output = self.attention.forward(
            query=query,
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=current_k,
            value=current_v
        )

        attn_output = attn_output.reshape(attn_output.size(0), -1)
        attn_output = layer.self_attn.o_proj(attn_output)
        attn_output = attn_output.unsqueeze(1)
        hidden_states = residual + attn_output

        # 前馈网络
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, (current_k, current_v)