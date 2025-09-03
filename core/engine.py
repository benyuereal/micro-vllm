# core/engine.py
import torch
from typing import List, Tuple, Dict
from .scheduler import Scheduler
from .cache_manager import KVCache
from .sequence import Sequence
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter
from transformers import DynamicCache
import torch.nn.functional as F



class InferenceEngine:

    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.cache = KVCache()
        self.adapter = QwenModelAdapter
        self.device = next(self.model.parameters()).device
        self.eos_token_id = self.tokenizer.eos_token_id  # ✅ 确保设置 eos_token_id

    def add_request(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9):
        seq_id = hash(prompt) % (2**32)
        seq = Sequence(seq_id, prompt, self.tokenizer, max_tokens)
        seq.temperature = temperature
        seq.top_p = top_p
        self.scheduler.add_request(seq)

    @torch.no_grad()
    def step(self) -> List[Tuple[int, List[int]]]:
        """
        执行一个推理 step，返回完成的序列
        """
        batch, batch_type = self.scheduler.get_next_batch()
        if not batch:
            return []

        seq_ids = [seq.seq_id for seq in batch]

        if batch_type == "prefill":
            # 合并 prefill 输入
            input_ids = [seq.get_next_input_ids() for seq in batch]
            input_tensor = self._pad_batch(input_ids, self.tokenizer.pad_token_id)
            input_tensor = input_tensor.to(self.device)

            # 执行prefill - 返回logits和DynamicCache
            logits, past_key_values = self._prefill_batch(input_tensor)
            print(f"past_key_values len: {len(past_key_values)}")
            print(f"layer 0 key shape: {past_key_values[0][0].shape}")  # [4, num_heads, seq_len, head_dim]
            print(f"layer 0 value shape: {past_key_values[0][1].shape}")

            # 拆分缓存
            new_kv_dict = self.cache.unbatch_kv(seq_ids, past_key_values)

            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                seq.update_state(next_token, new_kv_dict[seq.seq_id])

                if seq.is_finished():
                    self.scheduler.mark_finished(seq)
                else:
                    self.cache.allocate(seq.seq_id, seq.past_key_values, seq.current_position)

        elif batch_type == "decode":
            # 获取输入和 past_key_values
            input_ids = [seq.get_next_input_ids() for seq in batch]
            input_tensor = torch.tensor(input_ids, device=self.device)

            if not batch:
                return []
            # 获取 batch past_key_values
            batch_past_kv = self.cache.batch_kv(seq_ids)

            # 执行 decode
            logits, new_batch_kv = self._decode_batch(input_tensor, batch_past_kv)

            # 拆分并更新
            new_kv_dict = self.cache.unbatch_kv(seq_ids, new_batch_kv)
            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                seq.update_state(next_token, new_kv_dict[seq.seq_id])  # ✅ update_state 后立即检查 is_finished

                # ✅ 立即检查 is_finished（decode 后可能完成）
                if seq.is_finished():
                    self.cache.delete(seq.seq_id)
                    self.scheduler.mark_finished(seq)
                else:
                    self.cache.allocate(seq.seq_id, seq.past_key_values, seq.current_position)

        elif batch_type == "mixed":
            # 先 prefill，再 decode（复杂，建议先分开）
            # 这里可以进一步优化：使用 vLLM 的 PagedAttention 或 FlashInfer
            raise NotImplementedError("Mixed batch not implemented yet")

        # 返回空列表，因为完成序列已在mark_finished时处理
        return []

    def _prefill_batch(self, input_ids: torch.Tensor):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, None)
        outputs = self.model(**inputs)
        # 直接返回adapter处理后的DynamicCache
        return self.adapter.process_outputs(outputs, input_ids.size(1))


    def _decode_batch(self, input_ids: torch.Tensor, past_key_values: DynamicCache):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, past_key_values)
        outputs = self.model(**inputs)
        # 直接返回adapter处理后的DynamicCache
        return self.adapter.process_outputs(outputs, input_ids.size(1))


    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int):
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

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

    def generate(self, prompts: List[str], max_steps: int = 100):
        for prompt in prompts:
            self.add_request(prompt, max_steps)

        results = {}
        for step in range(max_steps):
            self.step()
            if not self.scheduler.running_sequences and not self.scheduler.waiting_queue:
                break

        # ✅ 关键：处理 finished_sequences 中的序列
        finished_from_scheduler = self.scheduler.get_finished_results()
        for seq, output_ids in finished_from_scheduler:
            try:
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                results[seq] = text
            except:
                results[seq] = f"[Error] {output_ids}"



        return results

