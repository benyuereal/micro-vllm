import torch
import threading
import time
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from dataclasses import dataclass
from .worker import ModelWorker
from .sampling.sampler import SamplingParams
from .utils.memory_utils import MemoryManager
from .schema import Request, Response
from .request_queue import RequestQueue  # 新增
from .response import batch_format_responses  # 新增


class LLMEngine:
    """LLM推理引擎，支持Continuous Batching"""

    def __init__(self,
                 model: str,
                 tokenizer: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 max_num_seqs: int = 256,
                 max_seq_length: int = 2048,
                 device: str = "cuda",
                 max_batch_size: int = 32):  # 添加批量大小参数
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

        # 使用RequestQueue管理请求
        self.request_queue_manager = RequestQueue(max_batch_size=max_batch_size)

        # 启动工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        print(f"LLMEngine initialized with model: {model}")

    def _worker_loop(self):
        """工作线程主循环（集成RequestQueue）"""
        while self.worker.is_running:
            try:
                # 获取下一个批次
                batch, response_queues = self.request_queue_manager.get_next_batch()
                if not batch:
                    time.sleep(0.01)
                    continue

                # 处理批次
                responses = self.worker.process_batch(batch)

                # 发送响应
                for response in responses:
                    self.request_queue_manager.send_response(response.request_id, response)
                    if response.success:
                        self.request_queue_manager.cleanup_request(response.request_id)

                time.sleep(0.001)

            except Exception as e:
                print(f"Error in worker loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def generate(self,
                 prompts: List[str],
                 sampling_params: SamplingParams,
                 as_json: bool = False) -> List[Any]:
        """生成文本（支持响应格式化）"""
        responses = []

        # 为每个提示创建请求
        for prompt in prompts:
            request_id = self.request_queue_manager.get_next_request_id()

            # 创建请求
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                arrival_time=time.time()
            )

            # 添加到请求队列
            self.request_queue_manager.add_request(request)

        # 获取批次并处理
        while True:
            batch, response_queues = self.request_queue_manager.get_next_batch()
            if not batch:
                break

            # 处理批次
            worker_responses = self.worker.process_batch(batch)

            # 发送响应
            for response in worker_responses:
                self.request_queue_manager.send_response(response.request_id, response)
                if response.success:
                    self.request_queue_manager.cleanup_request(response.request_id)
                    responses.append(response)

        # 格式化响应
        return batch_format_responses(responses, as_json=as_json)

    def shutdown(self):
        """关闭引擎"""
        # 通知工作线程退出
        self.worker.stop()
        self.worker_thread.join(timeout=5.0)