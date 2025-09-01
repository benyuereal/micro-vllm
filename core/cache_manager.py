# core/cache_manager.py (重构)
class KVCache:
    def __init__(self, max_batch_size=8):
        self.cache = {}
        self.max_batch_size = max_batch_size

    def allocate(self, seq_id: int, past_key_values: tuple):
        """为序列分配缓存空间"""
        self.cache[seq_id] = {
            'past_key_values': past_key_values,
            'active': True
        }

    def update(self, seq_id: int, past_key_values: tuple):
        """更新序列的缓存"""
        if seq_id in self.cache:
            self.cache[seq_id]['past_key_values'] = past_key_values

    def get_batch(self, seq_ids: list) -> tuple:
        """获取批量KV缓存"""
        batch_cache = []
        for seq_id in seq_ids:
            if seq_id in self.cache and self.cache[seq_id]['active']:
                batch_cache.append(self.cache[seq_id]['past_key_values'])
        return tuple(zip(*batch_cache)) if batch_cache else None

    def remove(self, seq_id: int):
        """移除序列缓存"""
        if seq_id in self.cache:
            del self.cache[seq_id]

    def free_inactive(self):
        """清理非活跃缓存"""
        to_remove = [seq_id for seq_id, cache in self.cache.items() if not cache['active']]
        for seq_id in to_remove:
            self.remove(seq_id)