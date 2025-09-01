import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .cache_manager import KVCache
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.cache = KVCache()
        self.adapter = QwenModelAdapter
        self.device = next(self.model.parameters()).device

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """处理初始提示（prefill阶段）"""
        inputs = self.adapter.prepare_inputs(self.model, input_ids, None)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def decode_step(self, input_ids: torch.Tensor, past_key_values: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """单步解码（decode阶段）"""
        inputs = self.adapter.prepare_inputs(
            self.model,
            input_ids,
            past_key_values
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def sample_next_token(self, logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """使用温度采样和top-p（核采样）选择下一个token"""
        # 应用温度
        logits = logits / temperature

        # 应用top-p（核采样）
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率低于top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        # 从剩余token中采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    def generate(self, prompts: list, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> dict:
        """生成文本"""
        results = {}

        # 预热阶段：处理初始提示
        for prompt in prompts:
            seq_id = id(prompt)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            logits, kv_cache = self.prefill(input_ids)

            # 分配缓存并存储初始KV
            self.cache.allocate(seq_id)
            self.cache.update(seq_id, kv_cache)

            # 获取最后一个token的预测
            next_token = self.sample_next_token(logits[0, -1, :], temperature, top_p)
            results[seq_id] = input_ids[0].tolist() + [next_token]

        # 解码循环 - 逐个序列处理
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
                if not cache_entry or cache_entry['past_key_values'] is None:
                    continue

                # 执行单序列解码
                input_tensor = torch.tensor([[last_token]], device=self.device)
                try:
                    logits, new_kv = self.decode_step(
                        input_tensor,
                        cache_entry['past_key_values']
                    )

                    # 获取下一个token
                    next_token = self.sample_next_token(logits[0, -1, :], temperature, top_p)
                    results[seq_id].append(next_token)

                    # 更新缓存
                    self.cache.update(seq_id, new_kv)

                except Exception as e:
                    print(f"Error decoding sequence {seq_id}: {str(e)}")
                    # 移除有问题的序列
                    if seq_id in results:
                        del results[seq_id]

            if not any_active:
                break

        # 结果转换
        decoded_results = {}
        for seq_id, token_ids in results.items():
            try:
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                # 移除可能的重复内容
                if decoded_text.startswith(prompts[0]):  # 简单去重逻辑
                    decoded_text = decoded_text[len(prompts[0]):]
                decoded_results[seq_id] = decoded_text
            except Exception as e:
                print(f"Error decoding sequence {seq_id}: {str(e)}")
                decoded_results[seq_id] = f"[Decoding error] Token IDs: {token_ids}"

        return decoded_results