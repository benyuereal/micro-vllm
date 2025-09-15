from typing import List, Sequence

import torch


def _process_decode_batch(self, batch: List[Sequence], seq_ids: List[int], context_lens: List[int]):
    """处理解码批次"""
    input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)

    token_positions = []
    for seq in batch:
        tokens = self.cache.get_sequence_tokens(seq.seq_id)
        positions = [min(pos, self.max_position - 1) for _, pos in tokens]
        token_positions.append(positions)

    # 修改这里：直接使用 self.model.transformer.wte 而不是 self.model.model.embed_tokens
    hidden_states = self.model.transformer.wte(input_ids)
    batch_size = len(input_ids)
    # 初始化列表以存储每层的Key和Value
    new_ks = []  # 每个元素形状: [batch_size, kv_num_heads, head_size]
    new_vs = []  # 每个元素形状: [batch_size, kv_num_heads, head_size]

    # 通过模型层
    for layer_idx, layer in enumerate(self.model.transformer.h):  # 注意这里也改为 transformer.h
        residual = hidden_states
        hidden_states = layer.ln_1(hidden_states)  # Qwen 使用 ln_1 而不是 input_layernorm

        # 计算Query、Key、Value
        query = layer.attn.c_attn(hidden_states)
        # Qwen 的 c_attn 是 q,k,v 的合并投影，需要拆分
        query, key, value = query.split(self.model.config.hidden_size, dim=2)

        # 获取头数和头尺寸
        head_size = self.paged_attention.head_size
        num_heads = self.paged_attention.num_heads
        kv_num_heads = self.paged_attention.kv_num_heads

        # 重塑Query、Key、Value
        query = query.view(batch_size, num_heads, head_size)
        key = key.view(batch_size, kv_num_heads, head_size)
        value = value.view(batch_size, kv_num_heads, head_size)

        # 保存Key和Value用于后续缓存
        new_ks.append(key)
        new_vs.append(value)

        # 获取当前token的位置（每个序列的当前位置）
        current_positions = [seq.current_position - 1 for seq in batch]

        # 调用PagedAttention
        attn_output = self.paged_attention.forward(
            query=query,
            cache_manager=self.cache,
            seq_ids=seq_ids,
            context_lens=context_lens,
            token_positions=token_positions,
            layer_idx=layer_idx,
            current_k=key,
            current_v=value,
            current_positions=current_positions
        )

        attn_output = attn_output.reshape(attn_output.size(0), -1)
        attn_output = layer.attn.c_proj(attn_output)  # Qwen 使用 c_proj 而不是 o_proj
        attn_output = attn_output.unsqueeze(1)
        hidden_states = residual + attn_output

        # 前馈网络
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)  # Qwen 使用 ln_2
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = self.model.transformer.ln_f(hidden_states)  # Qwen 使用 ln_f
    logits = self.model.lm_head(hidden_states).float()

    # 采样下一个token
    next_tokens = []
    for i, seq in enumerate(batch):
        next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        next_tokens.append(next_token)

    # 存储当前输入token的KV到缓存中
    for i, seq in enumerate(batch):
        input_token_id = input_ids[i].item()
        input_token_position = seq.current_position - 1

        num_layers = self.memory_pool.num_layers
        num_heads = self.memory_pool.num_heads
        head_size = self.memory_pool.head_size

        k_cache = torch.zeros((num_layers, num_heads, 1, head_size), dtype=self.memory_pool.dtype,
                              device=self.device)
        v_cache = torch.zeros_like(k_cache)

        for layer_idx in range(num_layers):
            k = new_ks[layer_idx][i]
            v = new_vs[layer_idx][i]
            k_cache[layer_idx] = k.view(1, num_heads, 1, head_size)
            v_cache[layer_idx] = v.view(1, num_heads, 1, head_size)

        self.cache.allocate(
            seq.seq_id,
            [(input_token_id, input_token_position)],
            k_cache,
            v_cache
        )

        self._update_sequence(seq, next_tokens[i])

@torch.no_grad()
def _process_prefill_batch(self, batch: List[Sequence], seq_ids: List[int], context_lens: List[int]):
    """处理预填充批次"""
    input_ids_list = [seq.get_next_input_ids() for seq in batch]
    input_tensor = self._pad_batch(input_ids_list, self.tokenizer.pad_token_id).to(self.device)

    # Prefill阶段使用标准模型实现
    outputs = self.model(input_ids=input_tensor, use_cache=True)
    logits, past_key_values = self.adapter.process_outputs(outputs, input_tensor.size(1))

    # 分配KV缓存
    for i, seq in enumerate(batch):
        # 提取序列的token和位置
        token_positions = []
        for pos, token_id in enumerate(seq.get_next_input_ids()):
            token_positions.append((token_id, pos))

        # 提取该序列的KV数据
        num_layers = self.memory_pool.num_layers
        num_heads = self.memory_pool.num_heads
        head_size = self.memory_pool.head_size
        seq_length = len(token_positions)

        # 初始化KV缓存张量
        k_cache = torch.zeros(
            (num_layers, num_heads, seq_length, head_size),
            dtype=self.memory_pool.dtype, device=self.device
        )
        v_cache = torch.zeros_like(k_cache)

        # 填充KV缓存 past_key_values.layers= 24 x [batch_size,num_heads, seq_length,head_size]
        for layer_idx in range(num_layers):
            layer_k = past_key_values[layer_idx][0][i:i + 1]  # [batch=1, num_heads, seq_len, head_size]
            layer_v = past_key_values[layer_idx][1][i:i + 1]

            k_cache[layer_idx] = layer_k.squeeze(0).permute(1, 0, 2)
            v_cache[layer_idx] = layer_v.squeeze(0).permute(1, 0, 2)

        # 为序列分配缓存 [num_layers,num_heads, seq_length,head_size]
        self.cache.allocate(seq.seq_id, token_positions, k_cache, v_cache)

        # 采样下一个token
        next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
        self._update_sequence(seq, next_token)