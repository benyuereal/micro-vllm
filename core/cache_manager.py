from typing import List


class KVCache:
    def __init__(self, max_length=2048):
        self.cache = {}
        self.max_length = max_length  # 最大缓存长度

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None,
            'length': 0
        }

    def update(self, seq_id, past_key_values):
        """更新缓存并记录长度"""
        if seq_id in self.cache:
            self.cache[seq_id]['past_key_values'] = past_key_values
            # 更新长度（取第一层key的长度）
            if past_key_values is not None:
                self.cache[seq_id]['length'] = past_key_values[0][0].shape[2]

    def get(self, seq_id):
        return self.cache.get(seq_id)

    def get_batch(self, seq_ids: List[int]):
        """批量获取缓存条目"""
        return [self.cache.get(id) for id in seq_ids]

    def prune(self, seq_id):
        """修剪过长的缓存"""
        if seq_id not in self.cache:
            return

        cache_entry = self.cache[seq_id]
        if cache_entry['length'] <= self.max_length:
            return

        # 修剪策略：保留最近的max_length个token
        new_cache = []
        for layer_cache in cache_entry['past_key_values']:
            key, value = layer_cache
            # 沿着序列维度切片
            pruned_key = key[:, :, -self.max_length:, :]
            pruned_value = value[:, :, -self.max_length:, :]
            new_cache.append((pruned_key, pruned_value))

        # 更新缓存
        cache_entry['past_key_values'] = tuple(new_cache)
        cache_entry['length'] = self.max_length