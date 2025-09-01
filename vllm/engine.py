import torch
import threading
import time
from typing import List, Optional, Tuple
from transformers import GenerationConfig

from config import Config
from request import Request
from scheduler import Scheduler
from cache_manager import KVCacheManager


class InferenceEngine:
    """推理引擎"""

    def __init__(self, model, tokenizer, kv_cache_manager: KVCacheManager, scheduler: Scheduler):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = scheduler
        self.running = True
        self.thread = threading.Thread(target=self.run_loop)
        self.thread.daemon = True

    def start(self):
        """启动推理引擎"""
        self.thread.start()
        print("Inference engine started")

    def stop(self):
        """停止推理引擎"""
        self.running = False
        self.thread.join()
        print("Inference engine stopped")

    def run_loop(self):
        """主推理循环"""
        while self.running:
            batch = self.scheduler.get_batch()
            if not batch:
                time.sleep(0.01)  # 无请求时休眠
                continue

            self.process_batch(batch)

    def process_batch(self, batch: List[Request]):
        """处理一批请求"""
        try:
            # 准备输入数据
            input_ids, attention_mask, position_ids, block_tables = self.prepare_batch_inputs(batch)

            # 运行模型推理
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True
                )

            # 处理输出
            next_token_logits = outputs.logits[:, -1, :]

            # 应用温度调节和top-p采样
            next_tokens = self.sample_next_tokens(next_token_logits, batch)

            # 更新请求状态
            self.update_requests(batch, next_tokens, outputs.past_key_values)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # 标记所有请求为完成状态
            for request in batch:
                request.finished = True
                self.scheduler.complete_request(request)

    def prepare_batch_inputs(self, batch: List[Request]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        """准备批次输入数据"""
        # 这里需要根据实际模型结构实现
        # 简化实现：仅返回占位符
        input_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=Config.DEVICE)
        attention_mask = torch.zeros((len(batch), 1), dtype=torch.long, device=Config.DEVICE)
        position_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=Config.DEVICE)
        block_tables = [req.block_ids for req in batch]

        return input_ids, attention_mask, position_ids, block_tables

    def sample_next_tokens(self, logits: torch.Tensor, batch: List[Request]) -> List[int]:
        """采样下一个token"""
        next_tokens = []
        for i, request in enumerate(batch):
            # 应用温度调节
            logits_i = logits[i] / request.temperature

            # 应用top-p筛选
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率高于top_p的token
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_i[indices_to_remove] = float('-inf')

            # 采样
            probs = torch.softmax(logits_i, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_tokens.append(next_token)

        return next_tokens

    def update_requests(self, batch: List[Request], next_tokens: List[int], past_key_values):
        """更新请求状态"""
        for i, request in enumerate(batch):
            token_id = next_tokens[i]
            request.output_tokens.append(token_id)
            request.remaining_tokens -= 1

            # 检查是否完成（遇到EOS或达到最大长度）
            if token_id == self.tokenizer.eos_token_id or request.remaining_tokens <= 0:
                request.finished = True
                self.scheduler.complete_request(request)

                # 解码并输出结果
                output_text = self.tokenizer.decode(request.output_tokens, skip_special_tokens=True)
                print(f"Request {request.request_id} completed: {output_text}")

            # 更新KV缓存（在实际实现中需要根据past_key_values更新）
            # 这里简化处理