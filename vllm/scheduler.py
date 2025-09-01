from typing import List, Deque, Dict
from collections import deque
import threading
from request import Request
from cache_manager import KVCacheManager


class Scheduler:
    """请求调度器"""

    def __init__(self, kv_cache_manager: KVCacheManager, max_batch_size: int = 4):
        self.kv_cache_manager = kv_cache_manager
        self.max_batch_size = max_batch_size
        self.request_queue: Deque[Request] = deque()
        self.active_requests: Dict[str, Request] = {}
        self.lock = threading.Lock()

    def add_request(self, request: Request) -> bool:
        """添加新请求"""
        with self.lock:
            # 计算需要的块数
            total_tokens = len(request.prompt) + request.max_tokens
            num_blocks = (total_tokens + self.kv_cache_manager.block_size - 1) // self.kv_cache_manager.block_size

            try:
                # 分配KV缓存块
                block_ids = self.kv_cache_manager.allocate_blocks(request.request_id, num_blocks)
                request.block_ids = block_ids
                self.request_queue.append(request)
                return True
            except RuntimeError as e:
                print(f"Failed to allocate blocks for request {request.request_id}: {e}")
                return False

    def get_batch(self) -> List[Request]:
        """获取一批待处理的请求"""
        with self.lock:
            batch = []
            while len(batch) < self.max_batch_size and self.request_queue:
                request = self.request_queue.popleft()
                if not request.finished:
                    batch.append(request)
                    self.active_requests[request.request_id] = request
            return batch

    def complete_request(self, request: Request):
        """完成请求并释放资源"""
        with self.lock:
            self.kv_cache_manager.free_blocks(request.request_id)
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

    def get_queue_size(self) -> int:
        """获取队列中的请求数量"""
        with self.lock:
            return len(self.request_queue)