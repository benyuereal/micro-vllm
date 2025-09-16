import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from core import KVCacheManager
from core.kv_store import KVStore

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
    def __init__(self, num_heads: int, head_size: int, kv_num_heads: int, device: str = "auto", kv_store: Optional[KVStore] = None):
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
            max_position=4096,  # 与模型的最大位置一致
            device=self.device
        )
        self.kv_store = kv_store
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
            query: torch.Tensor,  # [batch_size, num_heads, head_size]
            cache_manager: KVCacheManager,
            seq_ids: List[int],
            context_lens: List[int],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if True:
            return self._flash_attn_forward(
                query, cache_manager, seq_ids, context_lens, layer_idx, current_k, current_v
            )
        else:
            return self._compatible_forward(
                query, cache_manager, seq_ids, context_lens, layer_idx,
                current_k, current_v
            )

    def _flash_attn_forward(
            self,
            query: torch.Tensor,
            cache_manager: KVCacheManager,
            seq_ids: List[int],
            context_lens: List[int],
            layer_idx: int,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 准备块表 (batch_size, max_blocks)
        block_tables = []
        max_blocks = 0
        for seq_id in seq_ids:
            blocks = cache_manager.get_block_table(seq_id)
            block_tables.append(blocks)
            max_blocks = max(max_blocks, len(blocks))

        # 填充块表 (不足的补-1)
        padded_block_tables = []
        for blocks in block_tables:
            padded = blocks + [-1] * (max_blocks - len(blocks))
            padded_block_tables.append(padded)

        block_table_tensor = torch.tensor(
            padded_block_tables, dtype=torch.int32, device=self.device
        )

        # 获取KV缓存
        k_cache = cache_manager.get_k_cache(layer_idx)
        v_cache = cache_manager.get_v_cache(layer_idx)

        ## query、k的形状是:[batch_size, num_heads, head_size] 需要在第三个维度增加上seq_len满足旋转的参数需求
        positions = torch.tensor(
            context_lens, dtype=torch.int32, device=self.device
        ).unsqueeze(1)
        query = self.rotary_emb(query.unsqueeze(2), positions).squeeze(2)
        key = self.rotary_emb(key.unsqueeze(2), positions).squeeze(2)
        # 存储
        ##  进行存储
        for i, token_idx in enumerate(context_lens):
            seq_id = seq_ids[i]
            slot = cache_manager.append_token(seq_id, token_idx)
            if slot >= 0:
                # 存储经过旋转后的 k和未经旋转的v
                layer_kv = []
                k = key[i]
                v = value[i]
                layer_kv.append((k, v))
                # 存储到缓存
                print(f" decode layer kv shape, slot\f{slot}", k.shape)
                self.kv_store.store_tokens_layer_kv(layer_idx, [layer_kv], [slot])

        query = query.unsqueeze(1)
        # 使用flash_attn_with_kvcache
        output = flash_attn_with_kvcache(
            query,
            k_cache,
            v_cache,
            cache_seqlens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_table=block_table_tensor,
            softmax_scale=self.scale,
            causal=True
        )

        return output.squeeze(1)  # [batch_size, num_heads, head_size]

    def _compatible_forward(
            self,
            query: torch.Tensor,
            cache_manager: KVCacheManager,
            seq_ids: List[int],
            context_lens: List[int],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        macOS兼容实现，专门为Qwen 0.5B模型的GQA特性优化
        - Qwen 0.5B: 14个注意力头，2个KV头
        - 输入query形状: [batch_size, num_heads, head_size] (解码时)
        - 输出形状: [batch_size, num_heads, head_size]
        """
        batch_size = query.size(0)
        max_seq_len = max(context_lens) + (1 if current_k is not None else 0)

        # 准备KV缓存张量 [batch_size, kv_num_heads, max_seq_len, head_size]
        all_keys = torch.zeros(
            batch_size, self.kv_num_heads, max_seq_len, self.head_size,
            device=self.device, dtype=query.dtype
        )
        all_values = torch.zeros_like(all_keys)

        # 准备位置索引 [batch_size, max_seq_len]
        all_positions = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)

        # 从缓存中填充历史KV
        for i in range(batch_size):
            seq_id = seq_ids[i]
            seq_len = context_lens[i]
            positions = list(range(seq_len))

            # 获取slot映射
            slot_mapping = cache_manager.get_slot_mapping(seq_id, positions)

            # 获取当前层的KV缓存
            k_cache = cache_manager.get_k_cache(layer_idx)
            v_cache = cache_manager.get_v_cache(layer_idx)

            # 从缓存中提取KV
            for j, slot in enumerate(slot_mapping):
                if slot == -1:
                    continue

                block_id = slot // cache_manager.block_size
                pos_in_block = slot % cache_manager.block_size

                all_keys[i, :, j] = k_cache[block_id, pos_in_block]
                all_values[i, :, j] = v_cache[block_id, pos_in_block]
                all_positions[i, j] = positions[j]

        # 包括当前token（如果有）
        if current_k is not None and current_v is not None:
            # 将当前KV添加到序列末尾
            for i in range(batch_size):
                all_keys[i, :, context_lens[i]] = current_k[i]
                all_values[i, :, context_lens[i]] = current_v[i]
                all_positions[i, context_lens[i]] = context_lens[i]  # 当前位置
                context_lens[i] += 1


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