import torch
from typing import Dict, List, Tuple
from .cache_manager import KVCache
from .scheduler import Scheduler
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_model(model_path)
        self.cache = KVCache()
        self.adapter = QwenModelAdapter

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """处理初始提示（prefill阶段）"""
        inputs = self.adapter.prepare_inputs(self.model, input_ids)
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def decode_step(self, input_ids: torch.Tensor, past_key_values: Tuple, sequence_length: int) -> Tuple[
        torch.Tensor, Tuple]:
        """单步解码（decode阶段）"""
        inputs = self.adapter.prepare_inputs(
            self.model,
            input_ids,
            past_key_values,
            sequence_length
        )
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def generate(self, prompts: list, max_tokens: int = 128) -> dict:
        """生成文本"""
        results = {}
        sequence_info = {}  # 存储序列ID到元信息(长度，是否完成)的映射

        # 预热阶段：处理初始提示
        for prompt in prompts:
            seq_id = id(prompt)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            logits, kv_cache = self.prefill(input_ids)

            # 分配缓存并存储初始KV
            self.cache.allocate(seq_id)
            self.cache.update(seq_id, kv_cache, input_ids.size(1))

            # 获取最后一个token的预测
            next_token = logits[0, -1, :].argmax().item()
            results[seq_id] = input_ids[0].tolist() + [next_token]

            # 存储序列信息
            sequence_info[seq_id] = {
                "length": input_ids.size(1) + 1,
                "done": False
            }

        # 解码循环
        for step in range(max_tokens - 1):
            # 准备批处理输入
            input_batch = []
            cache_entries = []
            sequence_lengths = []
            prompt_ids = []

            # 收集所有未完成的序列
            for seq_id, data in sequence_info.items():
                if data["done"]:
                    continue

                # 获取最后一个生成的token
                last_token = results[seq_id][-1]
                input_batch.append(last_token)

                # 获取缓存
                cache_entry = self.cache.get(seq_id)
                cache_entries.append(cache_entry['past_key_values'])
                sequence_lengths.append(cache_entry['length'])
                prompt_ids.append(seq_id)

            if not input_batch:
                break

            # 执行批处理解码
            input_tensor = torch.tensor([input_batch], device="cuda")
            logits, new_kv = self.decode_step(
                input_tensor,
                cache_entries,
                sequence_lengths[0]  # 使用第一个序列的长度，因为Qwen所有序列需要相同长度
            )

            # 更新缓存和结果
            next_tokens = logits[0, -1, :].argmax(dim=-1).tolist()
            for i, seq_id in enumerate(prompt_ids):
                token_id = next_tokens[i]
                results[seq_id].append(token_id)

                # 更新序列长度
                self.cache.update(seq_id, new_kv[i], sequence_info[seq_id]["length"])
                sequence_info[seq_id]["length"] += 1

                # 检查是否生成了结束符
                if token_id == self.tokenizer.eos_token_id:
                    sequence_info[seq_id]["done"] = True

        # 结果转换
        return {k: self.tokenizer.decode(v) for k, v in results.items()}