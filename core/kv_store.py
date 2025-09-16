import torch
from typing import List, Tuple

from core import KVCacheManager
from core.cache_manager import store_kvcache


class KVStore:

    def __init__(self, cache_manager: KVCacheManager, block_size: int):
        """
        初始化KV存储管理器
        :param cache_manager: KV缓存管理器实例
        :param block_size: 缓存块大小
        """
        self.cache_manager = cache_manager
        self.block_size = block_size

    def store_layer_kv(self, layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor,
                              slot_mapping: List[int]):
        """
        存储单个层的KV缓存

        :param layer_idx: 层索引
        :param k_tensor: K张量，形状为 [num_tokens, num_kv_heads, head_dim]
        :param v_tensor: V张量，形状为 [num_tokens, num_kv_heads, head_dim]
        :param slot_mapping: 槽位映射列表，每个元素表示token在缓存中的位置
        """
        # 获取当前层的缓存
        k_cache, v_cache = self.cache_manager.get(layer_idx)

        # 将slot_mapping转为tensor
        slot_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device=k_tensor.device)

        # 存储到缓存
        store_kvcache(k_tensor, v_tensor, k_cache, v_cache, slot_tensor, self.block_size)

    def store_all_layers_kv(self, token_kv: List[List[Tuple[torch.Tensor, torch.Tensor]]], slot_mapping: List[int]):
        """
        存储所有层的KV缓存（按层遍历存储）

        :param token_kv: 多层KV数据，结构为:
            token_kv[token_idx][layer_idx] = (k_tensor, v_tensor)
        :param slot_mapping: 槽位映射列表
        """
        if not token_kv:
            return

        num_layers = len(token_kv[0])
        num_tokens = len(token_kv)

        # 按层处理
        for layer_idx in range(num_layers):
            # 收集当前层的所有token的K和V
            k_list = []
            v_list = []
            for token_idx in range(num_tokens):
                k, v = token_kv[token_idx][layer_idx]
                # 确保形状正确 [num_kv_heads, head_dim]
                if k.dim() == 3:
                    k = k.squeeze(1)
                if v.dim() == 3:
                    v = v.squeeze(1)
                k_list.append(k)
                v_list.append(v)

            # 堆叠成三维张量: [num_tokens, num_kv_heads, head_dim]
            k_tensor = torch.stack(k_list, dim=0)
            v_tensor = torch.stack(v_list, dim=0)

            # 存储当前层
            self.store_layer_kv(layer_idx, k_tensor, v_tensor, slot_mapping)


    def store_tokens_layer_kv(self, layer_idx: int, token_kv: List[List[Tuple[torch.Tensor, torch.Tensor]]], slot_mapping: List[int]):
        """
        存储所有层的KV缓存（按层遍历存储）

        :param token_kv: 多层KV数据，结构为:
            token_kv[token_idx][layer_idx] = (k_tensor, v_tensor)
        :param slot_mapping: 槽位映射列表
        """
        if not token_kv:
            return

        num_tokens = len(token_kv)

        # 按层处理

        # 收集当前层的所有token的K和V
        k_list = []
        v_list = []
        for token_idx in range(num_tokens):
            k, v = token_kv[token_idx][0]  # 修改这里：使用token_idx索引
        #    确保形状正确 [num_kv_heads, head_dim]
            k_list.append(k)
            v_list.append(v)

        # 堆叠成三维张量: [num_tokens, num_kv_heads, head_dim]
        k_tensor = torch.stack(k_list, dim=0)
        v_tensor = torch.stack(v_list, dim=0)

        # 存储当前层
        self.store_layer_kv(layer_idx, k_tensor, v_tensor, slot_mapping)