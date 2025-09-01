import torch
import threading
import time
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from dataclasses import dataclass
from .worker import ModelWorker
from .sampling.sampler import SamplingParams
from .utils.memory_utils import MemoryManager
from .schema import Request, Response  # 从schema导入

# @dataclass
# class Request:
#     request_id: int
#     prompt: str
#     sampling_params: SamplingParams
#     arrival_time: float
#     priority: int = 0
#
#
# @dataclass
# class Response:
#     request_id: int
#     generated_text: str
#     success: bool
#     error_message: Optional[str] = None


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
        # 设备自动补全
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            print("Using CUDA device:", device)
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
        """工作线程主循环（修复版：支持连续批处理）"""
        # 活动请求池
        active_requests = {}

        while self.worker.is_running:
            try:
                # 添加新请求到活动池
                while not self.request_queue.empty():
                    req = self.request_queue.get_nowait()
                    req_id = req.request_id

                    # 初始化生成状态
                    if not hasattr(req, 'generated_token_ids'):
                        req.generated_token_ids = []
                        req.prompt_ids = self.worker.tokenizer.encode(
                            req.prompt, return_tensors="pt"
                        ).to(self.worker.device).squeeze(0)
                        req.remaining_tokens = req.sampling_params.max_tokens
                        req.is_completed = False
                        req.start_time = time.time()

                    # 添加到活动请求池
                    active_requests[req_id] = (req, self.response_queues[req_id])

                # 如果没有活动请求，短暂休眠
                if not active_requests:
                    time.sleep(0.01)
                    continue

                # 准备批次（最多max_num_seqs个请求）
                batch_requests = []
                for req, queue in active_requests.values():
                    if not req.is_completed:
                        batch_requests.append(req)
                        if len(batch_requests) >= self.max_num_seqs:
                            break

                # 处理批次
                if batch_requests:
                    responses = self.worker.process_batch(batch_requests)

                    # 处理响应并更新活动池
                    for response in responses:
                        req_id = response.request_id
                        if req_id in active_requests:
                            req, queue = active_requests[req_id]

                            # 发送响应
                            queue.put(response)

                            # 如果请求完成，从活动池移除
                            if response.success:
                                del active_requests[req_id]
                                if req_id in self.response_queues:
                                    del self.response_queues[req_id]

                # 短暂休眠以避免过度占用CPU
                time.sleep(0.001)

            except Exception as e:
                print(f"Error in worker loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

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