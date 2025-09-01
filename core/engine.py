import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .scheduler import Scheduler
from .cache_manager import KVCache
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.cache = KVCache()
        self.scheduler = Scheduler()
        self.device = next(self.model.parameters()).device
        self.adapter = QwenModelAdapter

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

    def prefill_batch(self, batch: List[Tuple[int, str]]) -> Dict[int, torch.Tensor]:
        """批量处理预填充请求"""
        inputs = []
        seq_lengths = []

        for seq_id, prompt in batch:
            # 编码时添加padding_side设置
            self.tokenizer.padding_side = "right"
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            seq_lengths.append(input_ids.shape[1])
            inputs.append((input_ids, None))
            self.cache.allocate(seq_id)

        # 使用修改后的适配器方法
        model_inputs = self.adapter.prepare_batch_inputs(self.model, inputs)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        results = self.adapter.process_batch_outputs(outputs, seq_lengths)

        output_dict = {}
        for i, (seq_id, _) in enumerate(batch):
            logits, kv_cache = results[i]
            self.cache.update(seq_id, kv_cache, seq_lengths[i])
            output_dict[seq_id] = logits
            self.scheduler.move_to_decode(seq_id)

        return output_dict

    def decode_batch(self, batch: List[int]) -> Dict[int, torch.Tensor]:
        """批量处理解码请求"""
        inputs = []
        seq_lengths = []

        for seq_id in batch:
            cache_entry = self.cache.get(seq_id)
            if not cache_entry:
                continue

            # 输入是上一个生成的token
            last_token = torch.tensor(
                [[cache_entry['last_position']]],
                device=self.device
            )
            inputs.append((last_token, cache_entry['past_key_values']))
            seq_lengths.append(cache_entry['last_position'] + 1)

        model_inputs = self.adapter.prepare_batch_inputs(self.model, inputs)

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        results = self.adapter.process_batch_outputs(outputs, seq_lengths)

        output_dict = {}
        for i, seq_id in enumerate(batch):
            if i >= len(results):
                break

            logits, kv_cache = results[i]
            self.cache.update(seq_id, kv_cache, seq_lengths[i])
            output_dict[seq_id] = logits

        return output_dict
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

    def generate(self, prompts: list, max_tokens: int = 128, **kwargs) -> dict:
        """连续批处理生成接口"""
        # 初始化请求
        for prompt in prompts:
            self.scheduler.add_request(id(prompt), prompt)

        results = {id(prompt): [] for prompt in prompts}

        # 主处理循环
        while self.scheduler.has_pending_requests() and max_tokens > 0:
            # 获取当前批次
            batch = self.scheduler.get_batch()
            prefill_batch = batch['prefill']
            decode_batch = batch['decode']

            # 处理预填充批次
            if prefill_batch:
                prefill_outputs = self.prefill_batch(prefill_batch)
                for seq_id, logits in prefill_outputs.items():
                    next_token = self.sample_next_token(logits[0, -1, :], **kwargs)
                    results[seq_id].append(next_token)

            # 处理解码批次
            if decode_batch:
                decode_outputs = self.decode_batch(decode_batch)
                for seq_id, logits in decode_outputs.items():
                    next_token = self.sample_next_token(logits[0, -1, :], **kwargs)
                    results[seq_id].append(next_token)

                    # 检查结束条件
                    if next_token == self.tokenizer.eos_token_id:
                        self.scheduler.mark_finished(seq_id)
                        self.cache.free(seq_id)

            max_tokens -= 1

        # 转换结果为文本
        decoded_results = {}
        for seq_id, tokens in results.items():
            full_tokens = self.tokenizer.encode(prompts[0]) + tokens
            decoded_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
            decoded_results[seq_id] = decoded_text[len(prompts[0]):]  # 移除提示部分

        return decoded_results