# vllm/request_queue.py
import time
from queue import PriorityQueue, Queue
from typing import List, Optional, Tuple, Dict

from .schema import Request, Response


class RequestQueue:
    """LLM请求队列管理"""

    def __init__(self, max_batch_size: int = 8):
        self.request_queue = PriorityQueue()
        self.response_queues: Dict[int, Queue] = {}
        self.max_batch_size = max_batch_size
        self.request_counter = 0

    def add_request(self, request: Request):
        """添加新的请求到队列"""
        # 优先级计算：高优先级排在前面
        priority = (request.priority, -request.arrival_time)
        self.request_queue.put((priority, request))
        self.response_queues[request.request_id] = Queue()

    def get_next_batch(self) -> Tuple[List[Request], List[Queue]]:
        """获取下一个处理批次的请求和响应队列"""
        batch = []
        response_queues = []

        while not self.request_queue.empty() and len(batch) < self.max_batch_size:
            _, request = self.request_queue.get()
            batch.append(request)
            response_queues.append(self.response_queues[request.request_id])

        return batch, response_queues

    def send_response(self, request_id: int, response: Response):
        """发送响应到对应的响应队列"""
        if request_id in self.response_queues:
            self.response_queues[request_id].put(response)

    def cleanup_request(self, request_id: int):
        """清理完成的请求"""
        if request_id in self.response_queues:
            # 确保响应队列为空
            try:
                while not self.response_queues[request_id].empty():
                    self.response_queues[request_id].get_nowait()
            except:
                pass

            # 移除响应队列
            del self.response_queues[request_id]

    def get_next_request_id(self) -> int:
        """获取下一个请求ID"""
        self.request_counter += 1
        return self.request_counter