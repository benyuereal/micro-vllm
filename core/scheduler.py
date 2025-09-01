from collections import deque


class Scheduler:
    def __init__(self, max_batch_size=8):
        self.prefill_queue = deque()  # 等待预填充的新请求
        self.decode_queue = deque()  # 等待继续解码的请求
        self.max_batch_size = max_batch_size
        self.finished_sequences = set()

    def add_request(self, seq_id, prompt):
        """添加新请求到预填充队列"""
        self.prefill_queue.append((seq_id, prompt))

    def move_to_decode(self, seq_id):
        """将请求移动到解码队列"""
        self.decode_queue.append(seq_id)

    def mark_finished(self, seq_id):
        """标记请求已完成"""
        self.finished_sequences.add(seq_id)

    def get_batch(self):
        """获取当前批次的请求"""
        batch = {
            'prefill': [],
            'decode': []
        }

        # 获取预填充批次
        while self.prefill_queue and len(batch['prefill']) < self.max_batch_size:
            batch['prefill'].append(self.prefill_queue.popleft())

        # 获取解码批次
        while self.decode_queue and len(batch['decode']) < self.max_batch_size:
            seq_id = self.decode_queue.popleft()
            if seq_id not in self.finished_sequences:
                batch['decode'].append(seq_id)

        return batch

    def has_pending_requests(self):
        """检查是否有待处理请求"""
        return len(self.prefill_queue) > 0 or len(self.decode_queue) > 0