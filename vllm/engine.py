import torch
import threading
import time
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from dataclasses import dataclass
from .worker import ModelWorker
from .sampling.sampler import SamplingParams
from .utils.memory_utils import MemoryManager


@dataclass
class Request:
    request_id: int
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
    priority: int = 0


@dataclass
class Response:
    request_id: int
    generated_text: str
    success: bool
    error_message: Optional[str] = None


class LLMEngine:
    """LLM推理引擎，支持Continuous Batching"""

    def __init__(self,
                 model: str,
                 tokenizer: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 max_num_seqs: int = 256,
                 max_seq_length: int = 2048,
                 device: str = "cuda"):
        self.model_name = model
        self.tokenizer_name = tokenizer or model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self.max_seq_length = max_seq_length
        self.device = device

        # 初始化内存管理器
        self.memory_manager = MemoryManager()

        # 初始化模型工作器
        self.worker = ModelWorker(
            model_name=model,
            tokenizer_name=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_seq_length=max_seq_length,
            device=device,
            memory_manager=self.memory_manager
        )

        # 请求队列
        self.request_queue = Queue()
        self.response_queues = {}

        # 请求计数器
        self.request_counter = 0

        # 启动工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        print(f"LLMEngine initialized with model: {model}")

    def _worker_loop(self):
        """工作线程主循环"""
        while True:
            try:
                # 获取一批请求
                batch_requests = self._get_batch_requests()

                if batch_requests:
                    # 处理请求
                    responses = self.worker.process_batch(batch_requests)

                    # 将响应放入对应的队列
                    for response in responses:
                        # 统一处理字典和对象响应
                        req_id = response["request_id"] if isinstance(response, dict) else response.request_id
                        if req_id in self.response_queues:
                            self.response_queues[req_id].put(response)

                # 短暂休眠以避免过度占用CPU
                time.sleep(0.001)

            except Exception as e:
                print(f"Error in worker loop: {e}")
                time.sleep(0.1)  # 发生错误时短暂休眠

    def _get_batch_requests(self) -> List[Request]:
        """从队列中获取一批请求，实现Continuous Batching"""
        batch_requests = []

        try:
            # 获取第一个请求
            first_request = self.request_queue.get_nowait()
            batch_requests.append(first_request)

            # 尝试获取更多请求，直到达到批处理大小或队列为空
            while len(batch_requests) < self.max_num_seqs:
                request = self.request_queue.get_nowait()
                batch_requests.append(request)

        except Empty:
            pass  # 队列为空是正常情况

        return batch_requests

    def generate(self,
                 prompts: List[str],
                 sampling_params: SamplingParams) -> List[Response]:
        """生成文本"""
        responses = []
        response_queues = []

        # 为每个提示创建请求
        for prompt in prompts:
            self.request_counter += 1
            request_id = self.request_counter

            # 创建响应队列
            response_queue = Queue()
            self.response_queues[request_id] = response_queue
            response_queues.append(response_queue)

            # 创建请求
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                arrival_time=time.time()
            )

            # 将请求加入队列
            self.request_queue.put(request)

        # 等待所有响应
        for response_queue in response_queues:
            response = response_queue.get()
            responses.append(response)
            # 兼容字典和对象两种类型
            req_id = response['request_id'] if isinstance(response, dict) else response.request_id
            if req_id in self.response_queues:
                del self.response_queues[req_id]


        return responses

    def shutdown(self):
        """关闭引擎"""
        # 通知工作线程退出
        self.worker.stop()
        self.worker_thread.join(timeout=5.0)