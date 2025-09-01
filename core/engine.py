# core/engine.py
from typing import Tuple

import torch
from .cache_manager import KVCache
from .scheduler import Scheduler
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_model(model_path)
        self.cache = KVCache()
        self.scheduler = Scheduler()
        self.adapter = QwenModelAdapter

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """处理初始提示（prefill阶段）"""
        inputs = self.adapter.prepare_inputs(self.model, input_ids)
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def decode_step(self, input_ids: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """单步解码（decode阶段）"""
        inputs = self.adapter.prepare_inputs(self.model, input_ids, past_key_values)
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def generate(self, prompts: list, max_tokens: int = 128) -> dict:
        """生成文本"""
        results = {}

        # 预热阶段：处理初始提示
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            logits, kv_cache = self.prefill(input_ids)

            # 分配缓存并存储初始KV
            self.cache.allocate(id(prompt))
            self.cache.update(id(prompt), kv_cache)

            # 获取最后一个token的预测
            next_token = logits[0, -1, :].argmax().item()
            results[id(prompt)] = input_ids[0].tolist() + [next_token]

            # 将请求加入调度器
            self.scheduler.add_request(id(prompt), torch.tensor([[next_token]], device="cuda"))

        # 解码循环
        for _ in range(max_tokens - 1):
            batch = self.scheduler.get_batch()
            if not batch:
                break

            # 准备批处理输入
            input_batch = torch.cat([input_ids for _, input_ids in batch], dim=0)
            cache_entries = [self.cache.get(seq_id) for seq_id, _ in batch]

            # 执行批处理解码
            logits, new_kv = self.decode_step(input_batch, cache_entries)

            # 更新缓存和结果
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            for i, (seq_id, _) in enumerate(batch):
                token_id = next_tokens[i].item()
                results[seq_id].append(token_id)
                self.cache.update(seq_id, new_kv[i])
                self.scheduler.add_request(seq_id, torch.tensor([[token_id]], device="cuda"))

        # 结果转换
        return {k: self.tokenizer.decode(v) for k, v in results.items()}