import torch
from typing import List, Dict, Tuple, Optional
from collections import deque
import threading
from config import Config


class KVCacheBlock:
    """KV缓存块"""

    def __init__(self, block_id: int, block_size: int, num_layers: int, num_heads: int, head_dim: int, device: str,
                 dtype: torch.dtype):
        self.block_id = block_id
        self.k_cache = torch.zeros(
            (num_layers, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.occupied = False


class KVCacheManager:
    """KV缓存管理器"""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, block_size: int = None, max_blocks: int = None):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size or Config.BLOCK_SIZE
        self.max_blocks = max_blocks or Config.MAX_BLOCKS
        self.device = Config.DEVICE
        self.dtype = Config.DTYPE

        # 初始化块
        self.blocks = [
            KVCacheBlock(i, self.block_size, num_layers, num_heads, head_dim, self.device, self.dtype)
            for i in range(self.max_blocks)
        ]
        self.free_blocks = deque(range(self.max_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}
        self.lock = threading.Lock()

    def allocate_blocks(self, request_id: str, num_blocks: int) -> List[int]:
        """为请求分配块"""
        with self.lock:
            if len(self.free_blocks) < num_blocks:
                raise RuntimeError(
                    f"Not enough free blocks. Requested: {num_blocks}, Available: {len(self.free_blocks)}")

            allocated = []
            for _ in range(num_blocks):
                block_id = self.free_blocks.popleft()
                self.blocks[block_id].occupied = True
                allocated.append(block_id)

            self.allocated_blocks[request_id] = allocated
            return allocated

    def free_blocks(self, request_id: str):
        """释放请求占用的块"""
        with self.lock:
            if request_id in self.allocated_blocks:
                for block_id in self.allocated_blocks[request_id]:
                    self.blocks[block_id].occupied = False
                    self.free_blocks.append(block_id)
                del self.allocated_blocks[request_id]

    def get_kv_cache(self, block_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定块的KV缓存"""
        k_cache = torch.stack([self.blocks[bid].k_cache for bid in block_ids])
        v_cache = torch.stack([self.blocks[bid].v_cache for bid in block_ids])
        return k_cache, v_cache

    def update_kv_cache(self, block_id: int, layer_idx: int, k_values: torch.Tensor, v_values: torch.Tensor):
        """更新指定块的KV缓存"""
        self.blocks[block_id].k_cache[layer_idx].copy_(k_values)
        self.blocks[block_id].v_cache[layer_idx].copy_(v_values)