class KVCache:
    def __init__(self):
        self.cache = {}
        self.active_sequences = set()

    def allocate(self, seq_id):
        self.cache[seq_id] = {
            'past_key_values': None,
            'last_position': 0
        }
        self.active_sequences.add(seq_id)

    def update(self, seq_id, past_key_values, current_position):
        self.cache[seq_id]['past_key_values'] = past_key_values
        self.cache[seq_id]['last_position'] = current_position

    def get(self, seq_id):
        return self.cache.get(seq_id)

    def free(self, seq_id):
        """释放序列资源"""
        if seq_id in self.cache:
            del self.cache[seq_id]
            self.active_sequences.discard(seq_id)

    def has_active_sequences(self):
        return len(self.active_sequences) > 0