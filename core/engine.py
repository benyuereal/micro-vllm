import torch
from typing import Dict, List, Tuple
from .cache_manager import KVCache
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()  # 设置为评估模式
        self.cache = KVCache()
        self.adapter = QwenModelAdapter

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """处理初始提示（prefill阶段）"""
        inputs = self.adapter.prepare_inputs(self.model, input_ids)
        with torch.no_grad():  # 禁用梯度计算
            outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def decode_step(self, input_ids: torch.Tensor, past_key_values: Tuple, sequence_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple]:
        """单步解码（decode阶段）"""
        inputs = self.adapter.prepare_inputs(
            self.model,
            input_ids,
            past_key_values,
            sequence_lengths
        )
        with torch.no_grad():  # 禁用梯度计算
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
                # 验证缓存结构
                if cache_entry and 'past_key_values' in cache_entry:
                    cache_entries.append(cache_entry['past_key_values'])
                    sequence_lengths.append(cache_entry['length'])
                    prompt_ids.append(seq_id)
                else:
                    print(f"Warning: Missing cache for sequence {seq_id}")
                    sequence_info[seq_id]["done"] = True

            if not input_batch:
                break

            try:
                # 执行批处理解码
                input_tensor = torch.tensor([input_batch], device="cuda").T
                logits, new_kv = self.decode_step(
                    input_tensor,
                    cache_entries,
                    sequence_lengths
                )

                # 更新缓存和结果
                next_tokens = logits[:, -1, :].argmax(dim=-1).tolist()
                for i, seq_id in enumerate(prompt_ids):
                    token_id = next_tokens[i]
                    results[seq_id].append(token_id)

                    # 更新序列长度
                    new_length = sequence_info[seq_id]["length"] + 1
                    self.cache.update(seq_id, new_kv[i], new_length)
                    sequence_info[seq_id]["length"] = new_length

                    # 检查是否生成了结束符
                    if token_id == self.tokenizer.eos_token_id:
                        sequence_info[seq_id]["done"] = True
            except Exception as e:
                print(f"Error during decode step {step}: {str(e)}")
                # 将当前批次的所有序列标记为完成
                for seq_id in prompt_ids:
                    sequence_info[seq_id]["done"] = True

        # 结果转换
        decoded_results = {}
        for seq_id, token_ids in results.items():
            try:
                decoded_results[seq_id] = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"Error decoding sequence {seq_id}: {str(e)}")
                decoded_results[seq_id] = f"[Decoding error] Token IDs: {token_ids}"

        return decoded_results