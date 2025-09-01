class KVCache:
    def __init__(self):
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None
        }

    def update(self, seq_id, past_key_values):
        """更新缓存"""
        entry = self.cache[seq_id]
        entry['past_key_values'] = past_key_values

    def get(self, seq_id):
        return self.cache.get(seq_id)