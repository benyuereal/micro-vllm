from collections import deque


class Scheduler:
    def __init__(self):
        self.waiting_queue = deque()
        self.running_queue = deque()

    def add_request(self, request):
        self.waiting_queue.append(request)

    def schedule(self):
        if not self.waiting_queue:
            return None
        return self.waiting_queue.popleft()

    def get_running_requests(self):
        return list(self.running_queue)