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
        return self._flash_attn_forward(
            query, cache_manager, seq_ids, context_lens, layer_idx, current_k, current_v
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

        new_positions = [l + 1 for l in context_lens]  # 新token位置
        # 应用旋转位置编码
        positions = torch.tensor(
            new_positions, dtype=torch.int32, device=self.device
        ).unsqueeze(1)
        query = self.rotary_emb(query.unsqueeze(2), positions).squeeze(2)
        key = self.rotary_emb(key.unsqueeze(2), positions).squeeze(2)

        # 存储当前KV缓存（可能分配新块，改变块表）
        for i, token_idx in enumerate(context_lens):
            seq_id = seq_ids[i]
            # 由于获取的是最新的位置，那么就是按照当前长度来去查询
            slot = cache_manager.get_slot(seq_id, token_idx)
            # 添加调试日志
            print(f"Seq {seq_id} | Pos:  | Slot: {slot}")
            if slot >= 0:
                # 确保存储新token的KV
                k = key[i]
                v = value[i]
                self.kv_store.store_tokens_layer_kv(layer_idx, [[(k, v)]], [slot])

        # 在存储之后准备块表，因为存储可能分配了新块
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

        query = query.unsqueeze(1)
        # 使用flash_attn_with_kvcache
        updated_lens = [l + 1 for l in context_lens]  # 包含新token
        output = flash_attn_with_kvcache(
            query,
            k_cache,
            v_cache,
            cache_seqlens=torch.tensor(updated_lens, dtype=torch.int32, device=self.device),
            block_table=block_table_tensor,
            softmax_scale=self.scale,
            causal=True
        )

        return output.squeeze(1)  # [batch_size, num_heads, head_size]

