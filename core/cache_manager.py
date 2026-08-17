"""KVCacheManager：4D Block-Slot-Tensor paged KV 缓存管理。

KV 缓存张量形状 [num_blocks, block_size, num_heads, head_size]，每层一份。
1 Token → 1 Slot（slot_mapping 定位）；1 Block → block_size 个 Slot；
1 Sequence → 动态增长的多个 Block。分配/释放 O(1)（deque 空闲块列表）。
参考：vLLM / PagedAttention (arxiv 2309.06180)。
"""
from typing import List

import torch
import collections

try:
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    print('Please install flash-attn from https://www.flash-attn.org')


def is_macos():
    return torch.backends.mps.is_available()


# block_table 填充 kernel：一个线程写 block_table[i, j] 一个位置
if not is_macos() and torch.cuda.is_available():
    @triton.jit
    def _block_table_kernel(
            flat_blocks_ptr,  # [total_blocks] 所有的 block id 展平
            flat_offsets_ptr,  # [batch_size+1] 每个序列的起始偏移
            block_table_ptr,  # [max_batch, max_seq_blocks] 目标表
            batch_size: tl.constexpr,
            max_seq_blocks: tl.constexpr,
            BLOCK_TABLE_BATCH_STRIDE: tl.constexpr,
            BLOCK_TABLE_BLOCK_STRIDE: tl.constexpr,
    ):
        """一个线程负责写入 block_table 中的一个位置 (i, j)。"""
        idx = tl.program_id(0)
        i = idx // max_seq_blocks  # 当前处理的 batch index
        j = idx % max_seq_blocks  # 当前处理的 block index

        if i < batch_size:
            # 读取当前序列的范围 [start, end)
            start = tl.load(flat_offsets_ptr + i)
            end = tl.load(flat_offsets_ptr + i + 1)
            num_blocks = end - start

            # 如果 j 在范围内，写入数据；否则写入 -1
            if j < num_blocks:
                block_id = tl.load(flat_blocks_ptr + start + j)
            else:
                block_id = -1

            # 计算地址并写入
            offset = (i * BLOCK_TABLE_BATCH_STRIDE +
                      j * BLOCK_TABLE_BLOCK_STRIDE)
            tl.store(block_table_ptr + offset, block_id)


class KVCacheManager:
    """Paged KV 缓存管理器：alloc/append/free O(1)，按 seq_id 管理动态增长的 block 列表。"""

    def __init__(self,
                 n_blocks: int,  # 总Block数 (如1024)
                 block_size: int,  # 每个Block的Slot数 (如16)
                 n_layers: int,  # 模型层数 (如32)
                 n_heads: int,  # 注意力头数 (如16)
                 head_size: int,  # 每个头的维度 (如128)
                 dtype=torch.float16,  # 数据类型
                 device="cuda",
                 max_tokens: int = 1024,
                 max_batch_size: int = 32):
        self.n_blocks, self.block_size, self.n_layers = n_blocks, block_size, n_layers
        self.dtype, self.device = dtype, device

        # 每层一份 KV 缓存 [num_blocks, block_size, num_heads, head_size]
        self.k_caches, self.v_caches = [], []
        for _ in range(n_layers):
            shape = (n_blocks, block_size, n_heads, head_size)
            self.k_caches.append(torch.zeros(shape, dtype=dtype, device=device))
            self.v_caches.append(torch.zeros(shape, dtype=dtype, device=device))

        # 空闲块队列（deque O(1)）；seq_id → 已分配 blocks；block_id → 已用 slot 数
        self._free = collections.deque(range(n_blocks))
        self._blocks = {}
        self._pos = {}

        self.cache_seqlens = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.max_tokens = max_tokens
        self.max_seq_blocks = (max_tokens + block_size - 1) // block_size
        # 常驻静态缓冲区（graph 绑定，replay 时框架往里写真实表）
        self._block_table_buffer = torch.full(
            (max_batch_size, self.max_seq_blocks), -1, dtype=torch.int32, device=device)
        self._cache_seqlens_buffer = torch.zeros(max_batch_size, dtype=torch.int32, device=device)

        # block_table kernel 的输入暂存
        max_possible_blocks = max_batch_size * self.max_seq_blocks
        self._pre_blocks_buffer = torch.zeros(max_possible_blocks, dtype=torch.int32, device=device)
        self._offsets_buffer = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)

        # Pinned CPU staging（消除每步 torch.tensor() malloc）
        _pin = (device == "cuda")
        self._seqlens_cpu = torch.empty(max_batch_size, dtype=torch.int32, pin_memory=_pin)
        self._flat_cpu = torch.empty(max_possible_blocks, dtype=torch.int32, pin_memory=_pin)
        self._offsets_cpu = torch.empty(max_batch_size + 1, dtype=torch.int32, pin_memory=_pin)
        self._seqlens_np = self._seqlens_cpu.numpy()
        self._flat_np = self._flat_cpu.numpy()
        self._offsets_np = self._offsets_cpu.numpy()

        # 当前步分配了新 block 的 seq（decode 快速路径：dirty 才重建 block_table）
        self._dirty_seqs: set = set()

    def alloc(self, seq_id: int, n_tokens: int):
        """预填充阶段分配块，返回 (success, slot_mapping[n_tokens])。OOM 返回 (False, None)。"""
        n_needed = (n_tokens + self.block_size - 1) // self.block_size
        if len(self._free) < n_needed:
            return False, None

        blocks = [self._free.popleft() for _ in range(n_needed)]
        # 块位置计数器：除最后一块外都满，最后一块可能不满
        self._pos.update({
            b: n_tokens % self.block_size if i == len(blocks) - 1 else self.block_size
            for i, b in enumerate(blocks)
        })
        self._blocks[seq_id] = blocks

        # slot_mapping: token_idx → block_id * block_size + offset_in_block
        slot_mapping = torch.tensor([
            blocks[i // self.block_size] * self.block_size + i % self.block_size
            for i in range(n_tokens)
        ], dtype=torch.int32, device=self.device)
        return True, slot_mapping

    def append(self, seq_id: int):
        """解码阶段追加一个 token 的 slot（无可用块返回 -1）。"""
        if seq_id not in self._blocks:
            return -1

        blocks = self._blocks[seq_id]
        last_block = blocks[-1]
        current_pos = self._pos[last_block]

        if current_pos < self.block_size - 1:           # 当前块还有空间
            self._pos[last_block] += 1
            return last_block * self.block_size + current_pos
        elif self._free:                                # 分配新块
            new_block = self._free.popleft()
            blocks.append(new_block)
            self._pos[new_block] = 1
            self._dirty_seqs.add(seq_id)
            return new_block * self.block_size
        print(f"❌ cache_manager.append failed for seq {seq_id}, "
              f"blocks:{blocks} pos:{current_pos} bs:{self.block_size} free:{len(self._free)}")
        return -1

    def get(self, layer: int, block_id: int = None):
        """获取某层的 (k_cache, v_cache)；指定 block_id 则返回单个 block。"""
        k_cache = self.k_caches[layer]
        v_cache = self.v_caches[layer]
        return (k_cache[block_id], v_cache[block_id]) if block_id is not None else (k_cache, v_cache)

    def cache_batch_data(self, seq_ids: list, context_lens: list):
        batch_size = len(seq_ids)
        if batch_size == 0:
            return self._block_table_buffer[:0], self._cache_seqlens_buffer[:0]

        # 1. 收集并展平数据
        flat = []
        offsets = [0]
        for seq_id in seq_ids:
            blocks = self._blocks[seq_id]
            flat.extend(blocks)
            offsets.append(offsets[-1] + len(blocks))

        total_blocks = offsets[-1]

        # 2. 写入 context_lens（零分配：直写 pinned numpy → async H2D）
        self._seqlens_np[:batch_size] = context_lens
        self._cache_seqlens_buffer[:batch_size].copy_(self._seqlens_cpu[:batch_size], non_blocking=True)

        # 3. 批量写入辅助缓冲区
        self._flat_np[:total_blocks] = flat
        self._offsets_np[:batch_size + 1] = offsets
        self._pre_blocks_buffer[:total_blocks].copy_(self._flat_cpu[:total_blocks], non_blocking=True)
        self._offsets_buffer[:batch_size + 1].copy_(self._offsets_cpu[:batch_size + 1], non_blocking=True)

        # 4. 启动 Kernel 填充主表
        grid = (batch_size * self.max_seq_blocks,)
        _block_table_kernel[grid](
            self._pre_blocks_buffer,
            self._offsets_buffer,
            self._block_table_buffer,
            batch_size,
            self.max_seq_blocks,
            self._block_table_buffer.stride(0),
            self._block_table_buffer.stride(1),
        )

        return self._block_table_buffer[:batch_size], self._cache_seqlens_buffer[:batch_size]

    def prepare(self, seq_ids: list, context_lens: list, batch_switched: bool = False):
        """
        decode 每步缓存就绪检查：
        - batch 切换 或 有新 block 分配 → 全量重建（同时修正 seqlens 绝对值）
        - 否则完全跳过（block_table 不变，seqlens 由 commit GPU 原地维护）
        """
        if batch_switched or self._dirty_seqs:
            self.cache_batch_data(seq_ids, context_lens)
            self._dirty_seqs.clear()

    def commit(self, batch_size: int):
        """
        decode forward 完成后调用：GPU 原地将 cache_seqlens +1，无 H2D 拷贝。
        """
        self._cache_seqlens_buffer[:batch_size].add_(1)

    def free(self, seq_id: int):
        """释放 seq 的所有块回空闲队列（避免内存泄漏，必须调用）。"""
        if seq_id in self._blocks:
            for block_id in self._blocks[seq_id]:
                self._free.append(block_id)
                self._pos.pop(block_id, None)

            # 删除序列记录
            del self._blocks[seq_id]