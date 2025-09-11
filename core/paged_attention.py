import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class PagedAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, device: str = "auto"):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size ** 0.5)

        # 自动检测设备
        if device == "auto":
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        else:
            self.device = device

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
            query: torch.Tensor,  # [batch, num_heads, head_size]
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]]
    ) -> torch.Tensor:
        batch_size = query.size(0)
        output = torch.zeros_like(query)

        if self.use_cuda_kernel:
            return self._cuda_forward(query, cache_manager, seq_ids, context_lens, token_positions)
        else:
            return self._mps_forward(query, cache_manager, seq_ids, context_lens, token_positions)

    def _cuda_forward(
            self,
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]]
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
        all_k = torch.cat(all_k, dim=2)  # [layers, heads, total_slots, head_size]
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
            query: torch.Tensor,
            cache_manager: 'KVCache',
            seq_ids: List[int],
            context_lens: List[int],
            token_positions: List[List[int]]
    ) -> torch.Tensor:
        """MPS/CPU回退实现"""
        batch_size = query.size(0)
        output = torch.zeros_like(query)

        for i in range(batch_size):
            seq_id = seq_ids[i]
            tokens = cache_manager.get_sequence_tokens(seq_id)
            seq_len = context_lens[i]

            # 收集序列的所有KV
            keys = []
            values = []

            for token_id, position in tokens[:seq_len]:
                k, v = cache_manager.get_token_kv(token_id, position)
                keys.append(k)  # [layers, heads, head_size]
                values.append(v)

            # 拼接序列的KV
            K = torch.stack(keys, dim=2)  # [layers, heads, seq_len, head_size]
            V = torch.stack(values, dim=2)

            # 注意力计算
            q = query[i].unsqueeze(0)  # [1, num_heads, head_size]
            for layer_idx in range(K.size(0)):
                layer_k = K[layer_idx]  # [heads, seq_len, head_size]
                layer_v = V[layer_idx]

                # 计算注意力分数
                scores = torch.einsum("hd,sd->hs", q[0], layer_k) * self.scale
                attn = F.softmax(scores, dim=-1)

                # 计算加权和
                layer_output = torch.einsum("hs,hsd->hd", attn, layer_v)

                # 更新输出
                output[i] += layer_output

                # 为下一层调整query
                # (简化实现，实际模型中应包含线性变换)
                q = layer_output.unsqueeze(0)

        return output