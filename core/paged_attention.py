import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:

    print('flash_attn_with_kvcache not installed')


class RotaryEmbedding(nn.Module):
    # 保持原有实现不变
    def __init__(self, dim, max_position=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.device = device if device is not None else torch.device('cpu')
        # 使用指定数据类型创建 inv_freq
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device).to(torch.bfloat16) / dim))
        self.max_seq_len = max_position
        self._update_cos_sin_cache(max_position)

    def _update_cos_sin_cache(self, seq_len):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        self.sin_cache = emb.sin()[None, None, :, :]  # [1, 1, seq_len, dim]

    def forward(self, x, positions):
        """
        x: [batch_size, num_heads, seq_len, head_size]
        positions: [batch_size, seq_len] 位置索引
        """
        # 确保位置索引在正确的设备上
        positions = positions.to(self.device)

        # 确保缓存足够大
        max_pos = positions.max().item() + 1
        if max_pos > self.max_seq_len:
            self._update_cos_sin_cache(max_pos)

        # 获取对应的cos/sin值
        batch_size, num_heads, seq_len, head_size = x.shape

        # 展平位置索引并获取对应的cos/sin值
        positions_flat = positions.view(-1)  # [batch_size * seq_len]
        cos = self.cos_cache[:, :, positions_flat]  # [1, 1, batch_size * seq_len, dim]
        sin = self.sin_cache[:, :, positions_flat]  # [1, 1, batch_size * seq_len, dim]

        # 重塑为原始形状
        cos = cos.view(1, 1, batch_size, seq_len, head_size).permute(2, 1, 3, 0, 4).squeeze(
            3)  # [batch_size, num_heads, seq_len, head_size]
        sin = sin.view(1, 1, batch_size, seq_len, head_size).permute(2, 1, 3, 0, 4).squeeze(
            3)  # [batch_size, num_heads, seq_len, head_size]

        # 应用旋转位置编码
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]

        # 应用旋转公式
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)


class PagedAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, kv_num_heads: int, device: str = "auto"):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size ** 0.5)

        # 自动检测设备
        if device == "auto":
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 初始化旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            dim=head_size,
            max_position=4096,
            device=self.device
        )

        # 检查Flash Attention可用性
        self.use_flash_attn = False
        if self.device == 'cuda':
            try:
                from flash_attn import flash_attn_func, flash_attn_with_kvcache
                self.use_flash_attn = True
            except ImportError:
                self.use_flash_attn = False

    def forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCacheManager',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None,
            current_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        统一的注意力计算方法，自动选择最佳实现
        """
        if self.use_flash_attn:
            return self._flash_attn_forward(
                query, cache_manager, seq_ids, context_lens, token_positions,
                layer_idx, current_k, current_v, current_positions
            )
        else:
            return self._compatible_forward(
                query, cache_manager, seq_ids, context_lens, token_positions,
                layer_idx, current_k, current_v, current_positions
            )

    def _flash_attn_forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCacheManager',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None,
            current_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        A100 GPU上的高性能Flash Attention实现
        """


        # 准备块表
        block_tables = [cache_manager.get_block_table(seq_id) for seq_id in seq_ids]
        max_blocks = max(len(blocks) for blocks in block_tables) if block_tables else 0

        # 确保块表长度一致（不足的部分填充0）
        padded_block_tables = []
        for blocks in block_tables:
            padded = blocks + [0] * (max_blocks - len(blocks))
            padded_block_tables.append(padded)

        block_table_tensor = torch.tensor(padded_block_tables, dtype=torch.int32, device=self.device)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device=self.device)

        # 获取当前层的KV缓存
        k_cache = cache_manager.get_k_cache(layer_idx)
        v_cache = cache_manager.get_v_cache(layer_idx)

        print(f"Query shape: {query.shape}")
        print(f"K cache shape: {k_cache.shape}")
        print(f"V cache shape: {v_cache.shape}")
        print(f"block_table shape: {block_table_tensor.shape}")
        print(f"context_lens_tensor shape: {context_lens_tensor.shape}")

        # 使用Flash Attention
        output = flash_attn_with_kvcache(
            query.unsqueeze(1),  # 添加序列维度 [batch_size, 1, num_heads, head_size]
            k_cache, v_cache,
            cache_seqlens=context_lens_tensor,
            block_table=block_table_tensor,
            softmax_scale=self.scale,
            causal=True
        )

        return output.squeeze(1)  # 移除序列维度 [batch_size, num_heads, head_size]

    def _compatible_forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCacheManager',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None,
            current_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        macOS兼容的实现，使用标准PyTorch操作
        """
        batch_size = query.size(0)
        max_seq_len = max(context_lens) + 1 if current_k is not None else max(context_lens)

        # 获取缓存管理器参数
        block_size = cache_manager.block_size
        num_heads = cache_manager.num_heads
        head_size = cache_manager.head_size
        token_kv_size = num_heads * head_size  # 每个token在缓存中的大小

        # 准备批量KV缓存
        all_keys = torch.zeros(batch_size, self.kv_num_heads, max_seq_len, self.head_size,
                               device=self.device, dtype=query.dtype)
        all_values = torch.zeros_like(all_keys)

        # 准备位置编码
        all_positions = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)

        # 从缓存管理器中填充KV缓存
        for i in range(batch_size):
            seq_id = seq_ids[i]
            seq_len = context_lens[i]
            positions = token_positions[i]

            # 获取该序列的slot映射
            slot_mapping = cache_manager.get_slot_mapping(seq_id, positions[:seq_len])

            # 获取当前层的KV缓存
            k_cache = cache_manager.get_k_cache(layer_idx)
            v_cache = cache_manager.get_v_cache(layer_idx)

            # 从缓存中读取历史KV
            keys = []
            values = []
            for slot in slot_mapping:
                if slot == -1:
                    # 未找到的token，填充0
                    keys.append(torch.zeros(self.kv_num_heads, self.head_size,
                                            device=self.device, dtype=query.dtype))
                    values.append(torch.zeros(self.kv_num_heads, self.head_size,
                                              device=self.device, dtype=query.dtype))
                else:
                    # 关键修改：正确地从块中提取单个token的KV数据
                    # 计算块ID和在块内的偏移量
                    block_id = slot // block_size
                    offset_in_block = slot % block_size

                    # 计算在块内的起始位置
                    start_idx = offset_in_block * token_kv_size
                    end_idx = start_idx + (self.kv_num_heads * self.head_size)  # 实际KV数据大小

                    # 从块中提取对应的KV数据
                    k_flat = k_cache[block_id, start_idx:end_idx]
                    v_flat = v_cache[block_id, start_idx:end_idx]

                    # 重塑为正确的形状
                    k = k_flat.view(self.kv_num_heads, self.head_size)
                    v = v_flat.view(self.kv_num_heads, self.head_size)

                    keys.append(k)
                    values.append(v)

            # 包括当前token（如果有）
            if current_k is not None and current_v is not None and current_positions is not None:
                keys.append(current_k[i])
                values.append(current_v[i])
                positions = positions + [current_positions[i]]
                seq_len += 1

            # 填充到批量张量中
            if keys:
                K = torch.stack(keys, dim=1)  # [kv_num_heads, seq_len, head_size]
                V = torch.stack(values, dim=1)

                all_keys[i, :, :seq_len, :] = K
                all_values[i, :, :seq_len, :] = V
                all_positions[i, :seq_len] = torch.tensor(positions, device=self.device)

        # 应用旋转位置编码
        q_rot = query.unsqueeze(2)  # [batch_size, num_heads, 1, head_size]
        q_positions = all_positions[:, -1:] if current_k is not None else all_positions[:, -1:]
        q_rotated = self.rotary_emb(q_rot, q_positions)
        k_rotated = self.rotary_emb(all_keys, all_positions)

        # GQA处理: 确保键值头数与查询头数匹配
        if self.kv_num_heads != self.num_heads:
            repeat_times = self.num_heads // self.kv_num_heads
            k_rotated = k_rotated.repeat_interleave(repeat_times, dim=1)
            all_values = all_values.repeat_interleave(repeat_times, dim=1)

        # 计算注意力
        attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, all_values)

        return output.squeeze(2)  # [batch_size, num_heads, head_size]