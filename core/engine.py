# core/engine.py
import torch
from typing import List, Tuple, Dict
from .scheduler import Scheduler
from .cache_manager import KVCache
from .sequence import Sequence
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter

class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.cache = KVCache()
        self.adapter = QwenModelAdapter
        self.device = next(self.model.parameters()).device

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

        # 在 step() 开始
        print(f"Step: {len(self.scheduler.running_sequences)} running, {len(self.scheduler.waiting_queue)} waiting")

        # 在 decode 前
        for seq in batch:
            kv = self.cache.get(seq.seq_id)
            print(f"seq {seq.seq_id}: state={seq.state}, kv={kv is not None}, output_len={len(seq.output_ids)}")

        seq_ids = [seq.seq_id for seq in batch]

        if batch_type == "prefill":
            # 合并 prefill 输入
            input_ids = [seq.get_next_input_ids() for seq in batch]
            input_tensor = self._pad_batch(input_ids, self.tokenizer.pad_token_id)
            input_tensor = input_tensor.to(self.device)

            # 执行 prefill
            logits, past_key_values = self._prefill_batch(input_tensor)

            # 更新每个序列
            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                # 在 _prefill_batch 后
                new_kv_for_seq = tuple(layer[i:i + 1] for layer in past_key_values)  # ✅ tuple
                seq.update_state(next_token, new_kv_for_seq)
                # ✅ 即使完成也要存（至少存一次）
                if seq.seq_id not in self.cache.seq_kv_cache:
                    self.cache.allocate(seq.seq_id, seq.past_key_values)
                if seq.is_finished():
                    self.scheduler.mark_finished(seq)


        elif batch_type == "decode":
            # 获取输入和 past_key_values
            input_ids = [seq.get_next_input_ids() for seq in batch]
            input_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(-1)

            # 确保所有序列都有 past_key_values
            # 过滤掉没有 past_key_values 的序列
            valid_batch = []
            for seq in batch:
                kv = self.cache.get(seq.seq_id)
                if kv is None or len(kv) == 0:
                    print(f"[WARN] Skipping seq {seq.seq_id}: no past_key_values")
                    if seq.is_finished():
                        self.scheduler.mark_finished(seq)
                else:
                    valid_batch.append(seq)

            if not valid_batch:
                return []
            # 获取 batch past_key_values
            batch_past_kv = self.cache.batch_kv(seq_ids)

            # 执行 decode
            logits, new_batch_kv = self._decode_batch(input_tensor, batch_past_kv)

            # 拆分并更新
            new_kv_dict = self.cache.unbatch_kv(seq_ids, new_batch_kv)
            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                seq.update_state(next_token, new_kv_dict[seq.seq_id])
                if not seq.is_finished():
                    self.cache.allocate(seq.seq_id, seq.past_key_values)
                else:
                    self.cache.delete(seq.seq_id)

        elif batch_type == "mixed":
            # 先 prefill，再 decode（复杂，建议先分开）
            # 这里可以进一步优化：使用 vLLM 的 PagedAttention 或 FlashInfer
            raise NotImplementedError("Mixed batch not implemented yet")

        # 检查完成的序列
        finished = []
        for seq in batch:
            if seq.is_finished():
                self.scheduler.mark_finished(seq)
                finished.append((seq.seq_id, seq.output_ids))

        return finished

    def _prefill_batch(self, input_ids: torch.Tensor):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, None)
        outputs = self.model(**inputs)
        logits, past_key_values = self.adapter.process_outputs(outputs, input_ids.size(1))
        return logits, past_key_values

    def _decode_batch(self, input_ids: torch.Tensor, past_key_values: Tuple):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, past_key_values)
        outputs = self.model(**inputs)
        logits, new_past_key_values = self.adapter.process_outputs(outputs, input_ids.size(1))
        return logits, new_past_key_values

    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int):
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float):
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate(self, prompts: List[str], max_steps: int = 100):
        for prompt in prompts:
            self.add_request(prompt)

        results = {}
        for step in range(max_steps):
            finished = self.step()
            for seq_id, output_ids in finished:
                try:
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    results[seq_id] = text
                except:
                    results[seq_id] = f"[Error] {output_ids}"

            if not self.scheduler.running_sequences and not self.scheduler.waiting_queue:
                break

        return results
