class KVCache:
    def __init__(self, block_size=16, max_blocks=1024):
        self.block_size = block_size
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'k_blocks': [],
            'v_blocks': [],
            'length': 0
        }

    def update(self, seq_id, new_k, new_v):
        entry = self.cache[seq_id]
        entry['k_blocks'].append(new_k)
        entry['v_blocks'].append(new_v)
        entry['length'] += new_k.size(2)

    def get(self, seq_id):
        return self.cache.get(seq_id)