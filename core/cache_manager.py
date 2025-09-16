import torch



try:
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

except ImportError:
    print('flash_attn_with_kvcache not installed')


# macOS 兼容性处理
def is_macos():
    return torch.backends.mps.is_available()


# 核函数存储 KV Cache（macOS 使用 PyTorch 实现）
if not is_macos():

    @triton.jit
    def store_kvcache_kernel(
            key_ptr,
            key_stride_0,  # 第一维步长
            key_stride_1,  # 第二维步长
            key_stride_2,  # 第三维步长
            value_ptr,
            value_stride_0,
            value_stride_1,
            value_stride_2,
            k_cache_ptr,
            v_cache_ptr,
            slot_mapping_ptr,
            block_size: tl.constexpr,
            num_heads: tl.constexpr,
            head_size: tl.constexpr,
    ):
        # 获取当前token的索引
        token_idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + token_idx)
        if slot == -1:
            return

        # 计算块ID和在块内的偏移量
        block_id = slot // block_size
        offset_in_block = slot % block_size

        # 计算在块内的起始位置（以元素为单位）
        start_idx = offset_in_block * num_heads * head_size

        # 计算缓存中的起始位置（以块为单位）
        cache_base = block_id * block_size * num_heads * head_size

        # 处理每个头
        for head_idx in range(num_heads):
            # 计算key/value的内存偏移
            key_offset = token_idx * key_stride_0 + head_idx * key_stride_1
            value_offset = token_idx * value_stride_0 + head_idx * value_stride_1

            # 加载当前头的KV数据
            key = tl.load(key_ptr + key_offset + tl.arange(0, head_size))
            value = tl.load(value_ptr + value_offset + tl.arange(0, head_size))

            # 计算缓存中的位置
            cache_offset = cache_base + start_idx + head_idx * head_size + tl.arange(0, head_size)

            # 存储到缓存
            tl.store(k_cache_ptr + cache_offset, key)
            tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
        key: torch.Tensor,  # [num_tokens, num_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_heads, head_size]
        k_cache: torch.Tensor,  # [num_blocks, block_size * num_heads * head_size]
        v_cache: torch.Tensor,  # [num_blocks, block_size * num_heads * head_size]
        slot_mapping: torch.Tensor,  # [num_tokens]
        block_size: int,
        num_heads: int,
        head_size: int,
):
    """
    存储KV到缓存中，支持形状为[num_tokens, num_heads, head_size]的输入
    """
    num_tokens = key.size(0)

    # 检查输入形状
    assert key.dim() == 3 and value.dim() == 3
    assert key.shape == (num_tokens, num_heads, head_size)
    assert value.shape == (num_tokens, num_heads, head_size)
    assert slot_mapping.numel() == num_tokens

    if is_macos():
        # macOS兼容实现
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot == -1:
                continue

            # 计算块ID和在块内的偏移量
            block_id = slot // block_size
            offset_in_block = slot % block_size

            # 计算在块内的起始位置
            start_idx = offset_in_block * num_heads * head_size
            end_idx = start_idx + num_heads * head_size

            # 获取当前token的扁平化KV
            k_flat = key[i].reshape(-1)
            v_flat = value[i].reshape(-1)

            # 存储到缓存
            k_cache[block_id, start_idx:end_idx] = k_flat
            v_cache[block_id, start_idx:end_idx] = v_flat
    else:
        # CUDA高性能实现
        # 确保slot_mapping是正确类型的张量

        # 调用Triton内核
        store_kvcache_kernel[(num_tokens,)](
            key, key.stride(0), key.stride(1), key.stride(2),
            value, value.stride(0), value.stride(1), value.stride(2),
            k_cache, v_cache, slot_mapping,
            block_size, num_heads, head_size,
        )

# 重构的 KV Cache 管理类
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

        # 为每一层分配 KV 缓存
        self.k_caches = []
        self.v_caches = []
        for _ in range(num_layers):
            # 缓存形状: [num_blocks, block_size * num_heads * head_size]
            k_cache = torch.zeros((num_blocks, block_size * num_heads * head_size),
                                  dtype=dtype, device=device)
            v_cache = torch.zeros((num_blocks, block_size * num_heads * head_size),
                                  dtype=dtype, device=device)
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)

        # 空闲块管理
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}  # seq_id -> list of block indices
        self.block_tokens = {}  # block_id -> token positions

    def allocate(self, seq_id, num_tokens):
        """为序列分配缓存块并计算 slot 映射"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return False, None

        blocks = []
        slot_mapping = []
        for i in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            blocks.append(block_id)

            # 计算该块的 token 范围
            start_token = i * self.block_size
            end_token = min((i + 1) * self.block_size, num_tokens)

            # 记录块内的 token 位置
            self.block_tokens[block_id] = list(range(start_token, end_token))

            # 为每个 token 创建 slot 映射
            for token_idx in range(start_token, end_token):
                slot = block_id * self.block_size + (token_idx % self.block_size)
                slot_mapping.append(slot)

        self.allocated_blocks[seq_id] = blocks
        return True, torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

    def append_token(self, seq_id, token_position):
        """为序列追加 token 并返回 slot"""
        if seq_id not in self.allocated_blocks or not self.allocated_blocks[seq_id]:
            return -1

        # 获取最后一个块
        last_block_id = self.allocated_blocks[seq_id][-1]
        tokens_in_block = self.block_tokens[last_block_id]

        # 检查块是否已满
        if len(tokens_in_block) < self.block_size:
            # 块还有空间
            slot = last_block_id * self.block_size + len(tokens_in_block)
            tokens_in_block.append(token_position)
            return slot

        # 需要新块
        if not self.free_blocks:
            return -1

        new_block_id = self.free_blocks.pop(0)
        self.allocated_blocks[seq_id].append(new_block_id)
        self.block_tokens[new_block_id] = [token_position]
        return new_block_id * self.block_size

    def get_block_table(self, seq_id):
        """获取序列的块表"""
        return self.allocated_blocks.get(seq_id, [])

    def get_slot_mapping(self, seq_id, token_positions):
        """获取 token 位置的 slot 映射"""
        slot_mapping = []
        for pos in token_positions:
            for block_id in self.allocated_blocks.get(seq_id, []):
                if pos in self.block_tokens.get(block_id, []):
                    block_idx = self.block_tokens[block_id].index(pos)
                    slot = block_id * self.block_size + block_idx
                    slot_mapping.append(slot)
                    break
            else:
                slot_mapping.append(-1)  # 未找到

        return torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

    def deallocate(self, seq_id):
        """释放序列的缓存块"""
        if seq_id in self.allocated_blocks:
            for block_id in self.allocated_blocks[seq_id]:
                self.free_blocks.append(block_id)
                if block_id in self.block_tokens:
                    del self.block_tokens[block_id]
            del self.allocated_blocks[seq_id]

    def get_k_cache(self, layer_idx):
        return self.k_caches[layer_idx]

    def get_v_cache(self, layer_idx):
        return self.v_caches[layer_idx]