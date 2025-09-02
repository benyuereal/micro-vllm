# core/engine.py
import torch
from typing import List, Tuple, Dict
from .scheduler import Scheduler
from .cache_manager import KVCache
from .sequence import Sequence
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter
from transformers import DynamicCache


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
            print(f"past_key_values len: {len(past_key_values)}")
            print(f"layer 0 key shape: {past_key_values[0][0].shape}")  # [4, num_heads, seq_len, head_dim]
            print(f"layer 0 value shape: {past_key_values[0][1].shape}")

            # 更新每个序列
            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                # 在 _prefill_batch 后
                # ✅ 正确提取：对每个 layer 的 key 和 value 分别切片
                new_kv_for_seq = tuple(
                    (layer[0][i:i + 1], layer[1][i:i + 1])  # key: [i:i+1], value: [i:i+1]
                    for layer in past_key_values
                )

                # ✅ 检查是否提取成功（非空）
                if not new_kv_for_seq or len(new_kv_for_seq) == 0:
                    raise RuntimeError(f"Failed to extract kv for seq {seq.seq_id}, i={i}")
                seq.update_state(next_token, new_kv_for_seq)
                # ✅ 用 DynamicCache 包装
                if seq.seq_id not in self.cache.seq_kv_cache:
                    cache = DynamicCache.from_legacy_cache(seq.past_key_values)  # tuple → DynamicCache
                    self.cache.allocate(seq.seq_id, cache)  # 存入 DynamicCache
                if seq.is_finished():
                    self.scheduler.mark_finished(seq)


        elif batch_type == "decode":
            # 获取输入和 past_key_values
            input_ids = [seq.get_next_input_ids() for seq in batch]
            input_tensor = torch.tensor(input_ids, device=self.device)

            # 防御性检查：确保所有序列都是 decode 状态
            for seq in batch:
                if seq.state != "decode":
                    print(f"[ERROR] seq {seq.seq_id} is in {seq.state} state, expected decode")
                    raise RuntimeError(f"Invalid state in decode batch: {seq.state}")

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
            batch = valid_batch
            # 获取 batch past_key_values
            batch_past_kv = self.cache.batch_kv(seq_ids)

            # 执行 decode
            logits, new_batch_kv = self._decode_batch(input_tensor, batch_past_kv)

            # 拆分并更新
            new_kv_dict = self.cache.unbatch_kv(seq_ids, new_batch_kv)
            for i, seq in enumerate(batch):
                next_token = self.sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
                seq.update_state(next_token, new_kv_dict[seq.seq_id])
                new_kv = new_kv_dict[seq.seq_id]
                if new_kv is None:
                    print(f"[ERROR] seq {seq.seq_id} got None past_key_values from unbatch_kv")
                    raise RuntimeError(f"seq {seq.seq_id} got None past_key_values")
                if not seq.is_finished():
                    cache = DynamicCache.from_legacy_cache(seq.past_key_values)
                    self.cache.allocate(seq.seq_id, cache)
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

        # ✅ 用 DynamicCache 包装 tuple
        cache = DynamicCache.from_legacy_cache(past_key_values)  # tuple → DynamicCache
        return logits, cache  # 返回 cache 对象

    def _decode_batch(self, input_ids: torch.Tensor, past_key_values: "DynamicCache"):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, past_key_values)  # 传入 cache 对象
        outputs = self.model(**inputs)
        logits, new_past_key_values = self.adapter.process_outputs(outputs, input_ids.size(1))

        # ✅ new_past_key_values 已经是 DynamicCache 对象
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
