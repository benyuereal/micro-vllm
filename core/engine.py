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

    def decode_step(self, input_ids: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """单步解码（decode阶段）"""
        inputs = self.adapter.prepare_inputs(
            self.model,
            input_ids,
            past_key_values
        )
        with torch.no_grad():  # 禁用梯度计算
            outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def generate(self, prompts: list, max_tokens: int = 128) -> dict:
        """生成文本"""
        results = {}

        # 预热阶段：处理初始提示
        for prompt in prompts:
            seq_id = id(prompt)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            logits, kv_cache = self.prefill(input_ids)

            # 分配缓存并存储初始KV
            self.cache.allocate(seq_id)
            self.cache.update(seq_id, kv_cache)

            # 获取最后一个token的预测
            next_token = logits[0, -1, :].argmax().item()
            results[seq_id] = input_ids[0].tolist() + [next_token]

        # 解码循环 - 逐个序列处理，避免批处理问题
        for step in range(max_tokens - 1):
            any_active = False

            for prompt in prompts:
                seq_id = id(prompt)
                if seq_id not in results:
                    continue

                # 获取最后一个生成的token
                last_token = results[seq_id][-1]

                # 检查是否生成了结束符
                if last_token == self.tokenizer.eos_token_id:
                    continue

                any_active = True

                # 获取缓存
                cache_entry = self.cache.get(seq_id)
                if not cache_entry or 'past_key_values' not in cache_entry:
                    continue

                # 执行单序列解码
                input_tensor = torch.tensor([[last_token]], device="cuda")
                try:
                    logits, new_kv = self.decode_step(
                        input_tensor,
                        cache_entry['past_key_values']
                    )

                    # 获取下一个token
                    next_token = logits[0, -1, :].argmax().item()
                    results[seq_id].append(next_token)

                    # 更新缓存
                    self.cache.update(seq_id, new_kv)

                except Exception as e:
                    print(f"Error decoding sequence {seq_id}: {str(e)}")
                    # 移除有问题的序列
                    results.pop(seq_id, None)

            if not any_active:
                break

        # 结果转换
        decoded_results = {}
        for seq_id, token_ids in results.items():
            try:
                decoded_results[seq_id] = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"Error decoding sequence {seq_id}: {str(e)}")
                decoded_results[seq_id] = f"[Decoding error] Token IDs: {token_ids}"

        return decoded_results