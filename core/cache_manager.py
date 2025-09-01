class KVCache:
    def __init__(self):
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None,
            'length': 0  # 序列长度
        }

    def update(self, seq_id, past_key_values, length):
        """更新缓存和序列长度"""
        # 验证缓存结构
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple), "past_key_values must be a tuple"
            for i, layer_kv in enumerate(past_key_values):
                assert isinstance(layer_kv, tuple) and len(
                    layer_kv) == 2, f"Layer {i} should be a tuple of (key, value)"

        entry = self.cache[seq_id]
        entry['past_key_values'] = past_key_values
        entry['length'] = length

    def get(self, seq_id):
        return self.cache.get(seq_id)