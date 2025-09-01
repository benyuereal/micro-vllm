import torch


class KVCache:
    def __init__(self, block_size=16, max_blocks=1024):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache = {}

    def allocate_block(self, seq_id):
        if len(self.cache) >= self.max_blocks:
            raise RuntimeError("KV cache full")
        block_id = f"{seq_id}_{len(self.cache)}"
        self.cache[block_id] = {
            'k': torch.zeros((self.block_size, 4096), dtype=torch.float16),
            'v': torch.zeros((self.block_size, 4096), dtype=torch.float16),
            'free': True
        }
        return block_id

    def get_block(self, block_id):
        return self.cache[block_id]

    def free_block(self, block_id):
        self.cache[block_id]['free'] = True