# core/cache_manager.py
class KVCache:
    def __init__(self):
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None
        }

    def update(self, seq_id, past_key_values):
        """更新缓存"""
        self.cache[seq_id]['past_key_values'] = past_key_values

    def get(self, seq_id):
        return self.cache.get(seq_id)

    def remove(self, seq_id):
        """移除序列的缓存"""
        if seq_id in self.cache:
            del self.cache[seq_id]