import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.device = device if device is not None else torch.device('cpu')
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        self.max_seq_len = max_position
        self._update_cos_sin_cache(max_position)

    def _update_cos_sin_cache(self, seq_len):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cache = emb.cos()[None, None, :, :]
        self.sin_cache = emb.sin()[None, None, :, :]

    def forward(self, x, positions):
        """
        x: [batch_size, num_heads, seq_len, head_size]
        positions: [seq_len] 位置索引
        """
        # 确保位置索引在正确的设备上
        positions = positions.to(self.device)

        # 确保缓存足够大
        max_pos = positions.max().item() + 1
        if max_pos > self.max_seq_len:
            self._update_cos_sin_cache(max_pos)

        # 获取对应的cos/sin值
        cos = self.cos_cache[:, :, positions]
        sin = self.sin_cache[:, :, positions]

        # 应用旋转位置编码
        # 将x分成两部分
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
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        else:
            self.device = device

        # 初始化旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            dim=head_size,
            max_position=4096,  # 与模型的最大位置一致
            device=self.device
        )

        # 针对CUDA使用优化内核
        self.use_cuda_kernel = (self.device == 'cuda') and torch.cuda.is_available()
        if self.use_cuda_kernel:
            try:
                from vllm import paged_attention
                self.cuda_paged_attention = paged_attention
            except ImportError:
                print("vllm not available, using fallback implementation")
                self.use_cuda_kernel = False

    def forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int
    ) -> torch.Tensor:
        if self.use_cuda_kernel:
            return self._cuda_forward(query, cache_manager, seq_ids, context_lens, token_positions, layer_idx)
        else:
            return self._mps_forward(query, cache_manager, seq_ids, context_lens, token_positions, layer_idx)

    def _cuda_forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int
    ) -> torch.Tensor:
        """CUDA优化实现"""
        # 实现vLLM的块表格式
        max_blocks_per_seq = 0
        block_tables = []

        for seq_id in seq_ids:
            tokens = cache_manager.get_sequence_tokens(seq_id)
            block_table = []

            for token_id, position in tokens:
                allocation = cache_manager.memory_pool.find_token(token_id, position)
                if allocation:
                    block_id, slot_id = allocation
                    block_table.append(block_id * cache_manager.memory_pool.block_size + slot_id)

            block_tables.append(block_table)
            max_blocks_per_seq = max(max_blocks_per_seq, len(block_table))

        # 创建块表张量
        block_tables_tensor = torch.full(
            (len(seq_ids), max_blocks_per_seq),
            -1,
            dtype=torch.int, device=self.device
        )

        for i, table in enumerate(block_tables):
            if table:
                block_tables_tensor[i, :len(table)] = torch.tensor(table, device=self.device)

        # 获取所有KV块
        all_k = []
        all_v = []
        blocks = cache_manager.memory_pool.get_all_blocks()
        for block in blocks:
            all_k.append(block.key_cache)
            all_v.append(block.value_cache)

        # 拼接所有块 (vLLM要求连续内存)
        all_k = torch.cat(all_k, dim=2)  # [layers, kv_heads, total_slots, head_size]
        all_v = torch.cat(all_v, dim=2)

        # 调用vLLM内核
        return self.cuda_paged_attention(
            query,
            all_k, all_v,
            block_tables_tensor,
            torch.tensor(context_lens, device=self.device),
            self.scale
        )

    def _mps_forward(
            self,
            query: torch.Tensor,  # [batch_size, num_heads, head_size]
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]],
            layer_idx: int
    ) -> torch.Tensor:
        batch_size = query.size(0)
        output = torch.zeros_like(query)  # [batch_size, num_heads, head_size]

        for i in range(batch_size):
            seq_id = seq_ids[i]
            tokens = cache_manager.get_sequence_tokens(seq_id)
            seq_len = context_lens[i]

            if seq_len == 0:
                # 如果序列长度为0，输出零向量
                output[i] = 0
                continue

            # 收集 KV（注意：每个 token 的 k/v 是 [layers, kv_heads, head_size]）
            keys = []
            values = []
            for token_id, position in tokens[:seq_len]:
                k, v = cache_manager.get_token_kv(token_id, position)
                # 只取当前层的KV数据
                k = k[layer_idx]  # [kv_heads, head_size]
                v = v[layer_idx]
                keys.append(k)
                values.append(v)

            if not keys:
                # 如果没有token，输出零向量
                output[i] = 0
                continue

            # 拼接KV数据: [kv_heads, seq_len, head_size]
            K = torch.stack(keys, dim=1)  # [kv_heads, seq_len, head_size]
            V = torch.stack(values, dim=1)

            # 当前查询向量: [num_heads, head_size]
            q = query[i]  # [num_heads, head_size]

            # 获取位置信息（确保位置索引是LongTensor类型）
            positions = torch.tensor(
                token_positions[i][:seq_len],
                dtype=torch.long,
                device=self.device
            )

            # 应用旋转位置编码
            # 准备查询向量 [1, num_heads, 1, head_size]
            q_rot = q.unsqueeze(0).unsqueeze(2)
            # 准备键向量 [1, kv_heads, seq_len, head_size]
            k_rot = K.unsqueeze(0)

            # 应用RoPE
            q_rot = self.rotary_emb(q_rot, positions[-1:])
            k_rot = self.rotary_emb(k_rot, positions)

            # 更新查询和键
            q = q_rot.squeeze(0).squeeze(1)  # [num_heads, head_size]
            K = k_rot.squeeze(0)  # [kv_heads, seq_len, head_size]

            # GQA处理: 如果键值头数少于查询头数，重复键值头
            if K.size(0) != self.num_heads:
                # 计算重复次数
                repeat_times = self.num_heads // K.size(0)
                K = K.repeat_interleave(repeat_times, dim=0)  # [num_heads, seq_len, head_size]
                V = V.repeat_interleave(repeat_times, dim=0)

            # 注意力计算
            # q: [num_heads, head_size]
            # K: [num_heads, seq_len, head_size]
            # 计算点积: [num_heads, seq_len]
            scores = torch.einsum("hd,hsd->hs", q, K) * self.scale

            # 应用softmax
            attn = F.softmax(scores, dim=-1)

            # 加权值向量: [num_heads, head_size]
            layer_output = torch.einsum("hs,hsd->hd", attn, V)

            output[i] = layer_output

        return output