"""KVCacheManager：4D Block-Slot-Tensor paged KV 缓存管理。

KV 缓存张量形状 [num_blocks, block_size, num_heads, head_size]，每层一份。
1 Token → 1 Slot（slot_mapping 定位）；1 Block → block_size 个 Slot；
1 Sequence → 动态增长的多个 Block。分配/释放 O(1)（deque 空闲块列表）。
参考：vLLM / PagedAttention (arxiv 2309.06180)。
"""
import torch
import collections

try:
    import triton
    import triton.language as tl
except ImportError:
    print('Please install triton')


# 把本步算出的 k/v 按 slot_mapping scatter 写入 paged KV cache。
# slot_mapping[t] = block_id * block_size + offset_in_block，定位该 token 在
# [n_blocks*block_size, n_heads, head_size] 视图中的行。一个线程写一个 token 的一个 head。
@triton.jit
def _store_kvcache_kernel(
            k_ptr, v_ptr,          # [total_tokens, n_heads, head_size]
            k_stride_t, k_stride_h,# token 维 stride, head 维 stride
            v_stride_t, v_stride_h,
            kc_ptr, vc_ptr,        # paged cache [n_blocks*block_size, n_heads, head_size]
            slot_ptr,              # [total_tokens]
            N_HEAD: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            H_STRIDE: tl.constexpr,
    ):
        t = tl.program_id(0)       # token index
        slot = tl.load(slot_ptr + t)
        offs_h = tl.arange(0, N_HEAD)
        offs_d = tl.arange(0, HEAD_DIM)
        # 源：k[t, :, :] → [N_HEAD, HEAD_DIM]
        src_k = tl.load(k_ptr + t * k_stride_t + offs_h[:, None] * k_stride_h + offs_d[None, :])
        src_v = tl.load(v_ptr + t * v_stride_t + offs_h[:, None] * v_stride_h + offs_d[None, :])
        # 目的：cache[slot, :, :]
        dst = slot * H_STRIDE + offs_h[:, None] * HEAD_DIM + offs_d[None, :]
        tl.store(kc_ptr + dst, src_k)
        tl.store(vc_ptr + dst, src_v)


def store_kvcache(k, v, k_cache, v_cache, slot_mapping):
    """把 [total_tokens, n_heads, head_size] 的 k/v 按 slot_mapping 写入 paged cache。

    k/v 须 contiguous on last dim（head_dim）。slot_mapping[t] 定位 token t 在
    [n_blocks*block_size, n_heads, head_size] 视图中的行。"""
    total_tokens, n_heads, head_dim = k.shape
    kc = k_cache.reshape(-1, n_heads, head_dim)
    vc = v_cache.reshape(-1, n_heads, head_dim)
    _store_kvcache_kernel[(total_tokens,)](
        k, v, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        kc, vc, slot_mapping,
        N_HEAD=n_heads, HEAD_DIM=head_dim, H_STRIDE=kc.stride(0),
    )


# block_table 填充 kernel：一个线程写 block_table[i, j] 一个位置
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
        self._seqlens_cpu = torch.empty(max_batch_size, dtype=torch.int32, pin_memory=True)
        self._flat_cpu = torch.empty(max_possible_blocks, dtype=torch.int32, pin_memory=True)
        self._offsets_cpu = torch.empty(max_batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._seqlens_np = self._seqlens_cpu.numpy()
        self._flat_np = self._flat_cpu.numpy()
        self._offsets_np = self._offsets_cpu.numpy()

        # 当前步分配了新 block 的 seq（decode 快速路径：dirty 才重建 block_table）
        self._dirty_seqs: set = set()

        # prefix cache：满块前缀 hash → (block_id, token_chunk)。多请求共享前缀
        # （如 system prompt）时复用已算好的 KV block，跳过重复 prefill。
        # 只缓存满块（尾块不满不命中）；refcount 跟踪共享，归零才真正释放。
        self._prefix_cache: dict = {}
        self._refcount: dict = {}
        self._block_hash: dict = {}  # block_id → 前缀 hash（free 时 O(1) 移除表项）
        self._seq_hit_blocks: dict = {}  # seq_id → 命中复用的 block 集合（free 时只减这些）
        # refcount 语义：缓存自身持 1 份永久引用（register 时置 1），每个命中复用的
        # seq 各 +1。free 只减命中引用；归零（缓存引用也没了）才真正释放。

    def lookup_prefix(self, token_ids: list) -> int:
        """查 token 序列的最长已缓存满块前缀，返回命中 token 数（block_size 整数倍）。"""
        h = 0
        hit = 0
        for i in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            h = hash((h, tuple(token_ids[i:i + self.block_size])))
            if h not in self._prefix_cache:
                break
            hit += self.block_size
        return hit

    def _prefix_hashes(self, token_ids: list, n_tokens: int) -> list:
        """前 n_tokens 个 token（须为 block_size 整数倍）的链式前缀 hash 列表。"""
        hashes = []
        h = 0
        for i in range(0, n_tokens, self.block_size):
            h = hash((h, tuple(token_ids[i:i + self.block_size])))
            hashes.append(h)
        return hashes

    def alloc(self, seq_id: int, n_tokens: int, token_ids: list = None):
        """预填充阶段分配块，返回 (success, slot_mapping[n_tokens], prefix_hit)。

        prefix_hit = 命中的已缓存前缀 token 数（block_size 整数倍）。命中块复用
        （refcount+1，不重算），新块只覆盖 [prefix_hit, n_tokens)。
        调用方据 prefix_hit 设 seq.prefill_done，只 prefill 新 token。
        OOM 返回 (False, None, 0)。"""
        prefix_hit = 0
        prefix_blocks = []
        if token_ids is not None:
            prefix_hit = self.lookup_prefix(token_ids)
            if prefix_hit:
                prefix_blocks = [self._prefix_cache[h][0] for h in
                                 self._prefix_hashes(token_ids, prefix_hit)]
                for b in prefix_blocks:
                    self._refcount[b] = self._refcount.get(b, 0) + 1

        n_new = n_tokens - prefix_hit
        n_needed = (n_new + self.block_size - 1) // self.block_size
        if len(self._free) < n_needed:
            # 显存不足：先逐出无活跃 seq 引用的已缓存前缀块（refcount==1 仅缓存自身引用）
            for h, (b, _) in list(self._prefix_cache.items()):
                if len(self._free) >= n_needed:
                    break
                if self._refcount.get(b, 0) == 1:
                    del self._prefix_cache[h]
                    self._block_hash.pop(b, None)
                    self._refcount.pop(b, None)
                    self._free.append(b)
                    self._pos.pop(b, None)
            if len(self._free) < n_needed:
                # 仍不足：回滚已加的 refcount（前缀块留在缓存中，refcount=0 可被逐出）
                for b in prefix_blocks:
                    self._refcount[b] -= 1
                return False, None, 0

        blocks = [self._free.popleft() for _ in range(n_needed)]
        # 块位置计数器：除最后一块外都满，最后一块可能不满。
        # n_new 恰为 block_size 整数倍时最后一块也写满，_pos 须置 block_size
        # （而非 0）——否则 append() 误判该块有空位、复用 slot 0 且不分配新块，
        # 导致 block_table 少一列、flash 读 block_table[1]=-1 越界（illegal memory access）。
        last_pos = n_new % self.block_size or self.block_size
        self._pos.update({
            b: last_pos if i == len(blocks) - 1 else self.block_size
            for i, b in enumerate(blocks)
        })
        self._blocks[seq_id] = prefix_blocks + blocks
        self._seq_hit_blocks[seq_id] = set(prefix_blocks)

        # slot_mapping: 仅新 token [prefix_hit, n_tokens) → 新块 slot
        slot_mapping = torch.tensor([
            blocks[(i - prefix_hit) // self.block_size] * self.block_size
            + (i - prefix_hit) % self.block_size
            for i in range(prefix_hit, n_tokens)
        ], dtype=torch.int32, device=self.device)
        return True, slot_mapping, prefix_hit

    def register_prefix(self, seq_id: int, token_ids: list):
        """prefill 完成后登记本 seq 的满块前缀（供后续请求命中）。

        命中块 alloc 时已登记（在 _block_hash 中），跳过；新算的满块插入表并
        refcount+1（本 seq 持有）。"""
        blocks = self._blocks.get(seq_id)
        if not blocks:
            return
        n_full = len(token_ids) // self.block_size
        h = 0
        for i in range(0, n_full * self.block_size, self.block_size):
            h = hash((h, tuple(token_ids[i:i + self.block_size])))
            b = blocks[i // self.block_size]
            if b in self._block_hash:
                continue  # 命中块：alloc 时已登记且 refcount 已含本 seq
            self._prefix_cache[h] = (b, tuple(token_ids[i:i + self.block_size]))
            self._block_hash[b] = h
            # refcount = 缓存自身 1 份 + owner seq 1 份 = 2。owner 的引用记入
            # _seq_hit_blocks，free 时统一 -1（归到缓存引用 1，块留在表供后续命中）。
            self._refcount[b] = 2
            self._seq_hit_blocks.setdefault(seq_id, set()).add(b)

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
        """释放 seq 的所有块（避免内存泄漏，必须调用）。

        命中复用的 prefix 块：只减本 seq 的引用（refcount-1），缓存自身引用仍在
        （refcount>=1）→ 块留在 prefix 表供后续请求命中；refcount 归零（缓存引用
        也被逐出过）才真正回 _free。自有的新块直接回 _free。"""
        if seq_id in self._blocks:
            hit_blocks = self._seq_hit_blocks.pop(seq_id, set())
            for block_id in self._blocks[seq_id]:
                if block_id in hit_blocks:
                    self._refcount[block_id] = self._refcount.get(block_id, 1) - 1
                    if self._refcount[block_id] > 0:
                        continue  # 缓存引用（或其他 seq）仍在，保留
                    # 归零：缓存引用已不在（被逐出），真正释放
                    self._refcount.pop(block_id, None)
                    h = self._block_hash.pop(block_id, None)
                    if h is not None:
                        self._prefix_cache.pop(h, None)
                self._free.append(block_id)
                self._pos.pop(block_id, None)

            # 删除序列记录
            del self._blocks[seq_id]