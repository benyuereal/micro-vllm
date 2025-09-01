class KVCache:
    def __init__(self):
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None,
            'length': 0  # 新增序列长度跟踪
        }

    def update(self, seq_id, past_key_values, length):
        """更新缓存和序列长度"""
        entry = self.cache[seq_id]
        entry['past_key_values'] = past_key_values
        entry['length'] = length

    def get(self, seq_id):
        return self.cache.get(seq_id)