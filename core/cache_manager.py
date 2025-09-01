class KVCache:
    def __init__(self):
        self.cache = {}

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None
        }

    def update(self, seq_id, past_key_values):
        """更新缓存"""
        # 验证缓存结构
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple), "past_key_values must be a tuple"
            for i, layer in enumerate(past_key_values):
                assert isinstance(layer, tuple) and len(layer) == 2, f"Layer {i} should be a tuple of (key, value)"
                assert layer[0].dim() == 4 and layer[1].dim() == 4, f"Layer {i} keys and values should be 4D tensors"

        entry = self.cache[seq_id]
        entry['past_key_values'] = past_key_values

    def get(self, seq_id):
        return self.cache.get(seq_id)