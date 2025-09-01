from collections import deque


class Scheduler:
    def __init__(self, max_batch_size=4):
        self.queue = deque()
        self.max_batch_size = max_batch_size

    def add_request(self, seq_id, input_ids):
        self.queue.append((seq_id, input_ids))

    def get_batch(self):
        batch = []
        while self.queue and len(batch) < self.max_batch_size:
            batch.append(self.queue.popleft())
        return batch