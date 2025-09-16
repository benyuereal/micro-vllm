import torch
try:
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    print('Please install flash-attn from https://www.flash-attn.org')


def is_macos():
    return torch.backends.mps.is_available()


# Triton核函数用于高效存储KV缓存
if not is_macos() and torch.cuda.is_available():
    @triton.jit
    def store_kvcache_kernel(
            key_ptr,
            key_stride_b, key_stride_h, key_stride_d,
            value_ptr,
            value_stride_b, value_stride_h, value_stride_d,
            k_cache_ptr,
            v_cache_ptr,
            slot_mapping_ptr,
            block_size: tl.constexpr,
            num_heads: tl.constexpr,
            head_size: tl.constexpr,
            CACHE_BLOCK_STRIDE: tl.constexpr,
            CACHE_BLOCK_SIZE_STRIDE: tl.constexpr,
            CACHE_HEAD_STRIDE: tl.constexpr,
            CACHE_DIM_STRIDE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot == -1:
            return

        # 计算块ID和在块内的偏移量
        block_id = slot // block_size
        offset_in_block = slot % block_size

        # 计算输入key/value的偏移
        key_offset = (token_idx * key_stride_b +
                      head_idx * key_stride_h +
                      tl.arange(0, head_size) * key_stride_d)
        value_offset = (token_idx * value_stride_b +
                        head_idx * value_stride_h +
                        tl.arange(0, head_size) * value_stride_d)

        # 加载当前头的KV数据
        key = tl.load(key_ptr + key_offset)
        value = tl.load(value_ptr + value_offset)

        # 计算缓存中的位置
        cache_offset = (block_id * CACHE_BLOCK_STRIDE +
                        offset_in_block * CACHE_BLOCK_SIZE_STRIDE +
                        head_idx * CACHE_HEAD_STRIDE +
                        tl.arange(0, head_size) * CACHE_DIM_STRIDE)

        # 存储到缓存
        tl.store(k_cache_ptr + cache_offset, key)
        tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
        key: torch.Tensor,  # [num_tokens, num_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_heads, head_size]
        k_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
        v_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
        slot_mapping: torch.Tensor,  # [num_tokens]
        block_size: int,):
    num_tokens = key.size(0)
    num_heads = key.size(1)
    head_size = key.size(2)

    # 检查输入形状
    assert key.dim() == 3 and value.dim() == 3
    assert key.shape == (num_tokens, num_heads, head_size)
    assert value.shape == (num_tokens, num_heads, head_size)
    assert slot_mapping.numel() == num_tokens

    if is_macos() or not torch.cuda.is_available():
        # macOS兼容实现
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot == -1:
                continue

            block_id = slot // block_size
            offset_in_block = slot % block_size

            k_cache[block_id, offset_in_block] = key[i]
            v_cache[block_id, offset_in_block] = value[i]
    else:
        # CUDA高性能实现
        # 获取缓存步长
        CACHE_BLOCK_STRIDE = k_cache.stride(0)
        CACHE_BLOCK_SIZE_STRIDE = k_cache.stride(1)
        CACHE_HEAD_STRIDE = k_cache.stride(2)
        CACHE_DIM_STRIDE = k_cache.stride(3)

        # 调用Triton内核
        grid = (num_tokens, num_heads)
        store_kvcache_kernel[grid](
            key, key.stride(0), key.stride(1), key.stride(2),
            value, value.stride(0), value.stride(1), value.stride(2),
            k_cache, v_cache, slot_mapping,
            block_size, num_heads, head_size,
            CACHE_BLOCK_STRIDE, CACHE_BLOCK_SIZE_STRIDE,
            CACHE_HEAD_STRIDE, CACHE_DIM_STRIDE
        )


# 重构的KV缓存管理类（4D张量版本）
class KVCacheManager:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_size,
                 dtype=torch.float16, device="cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device

        # 为每一层分配KV缓存 [num_blocks, block_size, num_heads, head_size]
        self.k_caches = []
        self.v_caches = []
        for _ in range(num_layers):
            k_cache = torch.zeros(
                (num_blocks, block_size, num_heads, head_size),
                dtype=dtype, device=device
            )
            v_cache = torch.zeros(
                (num_blocks, block_size, num_heads, head_size),
                dtype=dtype, device=device
            )
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)

        # 空闲块管理
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}  # seq_id -> list of block indices
        self.block_positions = {}  # block_id -> 当前块内位置计数器

    def allocate(self, seq_id, num_tokens):
        """为序列分配缓存块并计算slot映射"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return False, None

        # 分配块
        blocks = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]

        # 初始化块位置计数器
        for block_id in blocks:
            self.block_positions[block_id] = 0

        self.allocated_blocks[seq_id] = blocks

        # 生成slot映射 [block_id, position_in_block]
        slot_mapping = []
        for i in range(num_tokens):
            block_idx = i // self.block_size
            pos_in_block = i % self.block_size
            block_id = blocks[block_idx]
            slot_mapping.append(block_id * self.block_size + pos_in_block)

        return True, torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

    def append_token(self, seq_id, token_position):
        """为序列追加token并返回slot"""
        if seq_id not in self.allocated_blocks:
            return -1

        blocks = self.allocated_blocks[seq_id]
        last_block = blocks[-1]
        current_pos = self.block_positions[last_block]

        if current_pos < self.block_size:
            # 当前块还有空间
            self.block_positions[last_block] += 1
            return last_block * self.block_size + current_pos
        elif self.free_blocks:
            # 需要新块
            new_block = self.free_blocks.pop(0)
            blocks.append(new_block)
            self.block_positions[new_block] = 0
            return new_block * self.block_size
        return -1

    def get_block_table(self, seq_id):
        """获取序列的块表"""
        return self.allocated_blocks.get(seq_id, [])

    def get_slot_mapping(self, seq_id, token_positions):
        """获取token位置的slot映射"""
        slot_mapping = []
        if seq_id not in self.allocated_blocks:
            return torch.tensor([-1] * len(token_positions), dtype=torch.int32, device=self.device)

        blocks = self.allocated_blocks[seq_id]
        for pos in token_positions:
            block_idx = pos // self.block_size
            pos_in_block = pos % self.block_size

            if block_idx < len(blocks):
                block_id = blocks[block_idx]
                slot = block_id * self.block_size + pos_in_block
                slot_mapping.append(slot)
            else:
                slot_mapping.append(-1)

        return torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

    def deallocate(self, seq_id):
        """释放序列的缓存块"""
        if seq_id in self.allocated_blocks:
            for block_id in self.allocated_blocks[seq_id]:
                self.free_blocks.append(block_id)
                if block_id in self.block_positions:
                    del self.block_positions[block_id]
            del self.allocated_blocks[seq_id]

    def get_k_cache(self, layer_idx):
        return self.k_caches[layer_idx]

    def get_v_cache(self, layer_idx):
        return self.v_caches[layer_idx]