import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

# 尝试导入flash_attn库，如果不可用则设为None
try:
    import flash_attn
    from flash_attn import flash_attn_func

    flash_attn_available = True
except ImportError:
    flash_attn_available = False

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


class FlashAttention(nn.Module):
    """Flash Attention实现，支持MPS和CPU后端"""

    def __init__(self, head_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(head_size)

    def forward(self, q, k, v, mask=None):
        """
        简化的Flash Attention实现
        q: [batch_size, num_heads, seq_len, head_size]
        k: [batch_size, num_heads, seq_len, head_size]
        v: [batch_size, num_heads, seq_len, head_size]
        mask: [batch_size, seq_len] 可选
        """
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用mask（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 应用softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用dropout
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 计算输出
        output = torch.matmul(attn_weights, v)

        return output


class OfficialFlashAttention(nn.Module):
    """真正的Flash Attention实现，使用flash_attn库"""

    def __init__(self, head_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(head_size)

    def forward(self, q, k, v, mask=None):
        """
        使用flash_attn库的实现
        q: [batch_size, num_heads, seq_len, head_size]
        k: [batch_size, num_heads, seq_len, head_size]
        v: [batch_size, num_heads, seq_len, head_size]
        mask: [batch_size, seq_len] 可选
        """
        # 转换mask格式为flash_attn需要的格式
        if mask is not None:
            # flash_attn需要bool类型的mask
            mask = mask.bool()

        # 使用flash_attn库的函数
        return flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                               softmax_scale=self.scale, causal=False, window_size=(-1, -1))


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
            max_position=4096,  # 与模型的最大位置一致
            device=self.device
        )

        # 根据设备和flash_attn可用性选择不同的Attention实现
        if flash_attn_available:
            # 在CUDA设备上且flash_attn可用时使用真正的Flash Attention
            self.flash_attn = OfficialFlashAttention(head_size)
            self.use_real_flash_attn = True
        else:
            # 在其他设备上或flash_attn不可用时使用模拟实现
            self.flash_attn = FlashAttention(head_size)
            self.use_real_flash_attn = False

    def forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,  # 当前token的K [batch_size, kv_num_heads, head_size]
            current_v: Optional[torch.Tensor] = None,  # 当前token的V [batch_size, kv_num_heads, head_size]
            current_positions: Optional[List[int]] = None  # 当前token的位置列表
    ) -> torch.Tensor:

        return self._flash_forward(
            query, cache_manager, seq_ids, context_lens, token_positions, layer_idx,
            current_k, current_v, current_positions
        )

    def _flash_forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int,
            current_k: Optional[torch.Tensor] = None,
            current_v: Optional[torch.Tensor] = None,
            current_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """原有的模拟Flash Attention实现"""
        batch_size = query.size(0)
        max_seq_len = max(context_lens) + 1 if current_k is not None else max(context_lens)

        # 准备批量KV缓存
        all_keys = torch.zeros(batch_size, self.kv_num_heads, max_seq_len, self.head_size,
                               device=self.device, dtype=query.dtype)
        all_values = torch.zeros_like(all_keys)

        # 准备位置编码
        all_positions = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)

        # 填充KV缓存和位置信息
        for i in range(batch_size):
            seq_id = seq_ids[i]
            tokens = cache_manager.get_sequence_tokens(seq_id)
            seq_len = context_lens[i]

            # 获取历史KV
            keys = []
            values = []
            positions_list = []

            for token_id, position in tokens[:seq_len]:
                k, v = cache_manager.get_token_kv(token_id, position)
                k = k[layer_idx]  # [kv_num_heads, head_size]
                v = v[layer_idx]
                keys.append(k)
                values.append(v)
                positions_list.append(position)

            # 包括当前token
            if current_k is not None and current_v is not None and current_positions is not None:
                keys.append(current_k[i])  # [kv_num_heads, head_size]
                values.append(current_v[i])
                positions_list.append(current_positions[i])
                seq_len += 1

            if keys:
                # 填充到批量张量中
                K = torch.stack(keys, dim=1)  # [kv_num_heads, seq_len, head_size]
                V = torch.stack(values, dim=1)

                all_keys[i, :, :seq_len, :] = K
                all_values[i, :, :seq_len, :] = V
                all_positions[i, :seq_len] = torch.tensor(positions_list, device=self.device)

        # 应用旋转位置编码
        # 调整query形状: [batch_size, num_heads, 1, head_size]
        q_rot = query.unsqueeze(2)

        # 应用RoPE到查询
        q_positions = all_positions[:, -1:] if current_k is not None else all_positions[:, -1:]
        q_rotated = self.rotary_emb(q_rot, q_positions)

        # 应用RoPE到键
        k_rotated = self.rotary_emb(all_keys, all_positions)

        print("kv_num_heads: ", self.kv_num_heads)
        print("num_heads: ", self.num_heads)
        # GQA处理: 确保键值头数与查询头数匹配
        if self.kv_num_heads != self.num_heads:

            # 确保查询头数是键值头数的整数倍
            if self.num_heads % self.kv_num_heads != 0:
                raise ValueError(f"num_heads({self.num_heads}) must be divisible by "
                                 f"kv_num_heads({self.kv_num_heads}) for Grouped Query Attention")

            repeat_times = self.num_heads // self.kv_num_heads
            k_rotated = k_rotated.repeat_interleave(repeat_times, dim=1)
            all_values = all_values.repeat_interleave(repeat_times, dim=1)

        # 使用Flash Attention
        # 调整形状: [batch_size, num_heads, seq_len, head_size]
        output = self.flash_attn(
            q_rotated,  # [batch_size, num_heads, 1, head_size]
            k_rotated,  # [batch_size, num_heads, max_seq_len, head_size]
            all_values,  # [batch_size, num_heads, max_seq_len, head_size]
        )

        return output.squeeze(2)  # [batch_size, num_heads, head_size]

