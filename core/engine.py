import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator
from queue import Queue
import time

from transformers import DynamicCache

from .memory_manager import MemoryPool
from .scheduler import Scheduler
from .cache_manager import KVCache
from .sequence import Sequence
from .model_loader import load_model
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()

        # 初始化内存池和KV缓存
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_size = self.model.config.hidden_size // num_heads
        num_key_value_heads = self.model.config.num_key_value_heads
        # 获取模型的数据类型
        model_dtype = next(self.model.parameters()).dtype

        self.memory_pool = MemoryPool(
            block_size=1024,
            max_blocks=32,
            num_layers=num_layers,
            head_size=head_size,
            num_heads=num_key_value_heads,
            dtype=model_dtype  # 使用模型的数据类型
        )

        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.cache = KVCache(self.memory_pool)
        self.adapter = QwenModelAdapter
        self.device = next(self.model.parameters()).device
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stream_callbacks = {}

    def add_request(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> int:
        seq_id = hash(prompt) % (2 ** 32)
        seq = Sequence(seq_id, prompt, self.tokenizer, max_tokens)
        seq.temperature = temperature
        seq.top_p = top_p
        self.scheduler.add_request(seq)
        return seq_id

    def register_stream_callback(self, seq_id: int, callback):
        self.stream_callbacks[seq_id] = callback

    def unregister_stream_callback(self, seq_id: int):
        self.stream_callbacks.pop(seq_id, None)

    def _invoke_stream_callback(self, seq_id: int, token: int, text: str):
        if callback := self.stream_callbacks.get(seq_id):
            try:
                callback(token, text)
            except Exception as e:
                print(f"Error in stream callback for seq {seq_id}: {e}")

    @torch.no_grad()
    def step(self) -> bool:
        """执行一个推理step，返回是否有处理任何请求"""
        batch, batch_type = self.scheduler.get_next_batch()
        if not batch:
            return False

        seq_ids = [seq.seq_id for seq in batch]

        if batch_type == "prefill":
            self._process_prefill_batch(batch, seq_ids)
        elif batch_type == "decode":
            self._process_decode_batch(batch, seq_ids)

        return True

    def _process_prefill_batch(self, batch: List[Sequence], seq_ids: List[int]):
        input_ids = [seq.get_next_input_ids() for seq in batch]
        input_tensor = self._pad_batch(input_ids, self.tokenizer.pad_token_id).to(self.device)

        logits, past_key_values = self._prefill_batch(input_tensor)
        new_kv_dict = self.cache.unbatch_kv(seq_ids, past_key_values)

        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            self._update_sequence(seq, next_token, new_kv_dict[seq.seq_id])

    def _process_decode_batch(self, batch: List[Sequence], seq_ids: List[int]):
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)
        batch_past_kv = self.cache.batch_kv(seq_ids)

        logits, past_key_values = self._decode_batch(input_ids, batch_past_kv)
        new_kv_dict = self.cache.unbatch_kv(seq_ids, past_key_values)

        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            self._update_sequence(seq, next_token, new_kv_dict[seq.seq_id])

    def _update_sequence(self, seq: Sequence, next_token: int, new_kv: DynamicCache):
        """更新序列状态"""
        seq.update_state(next_token, new_kv)

        # 流式输出回调
        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        self._invoke_stream_callback(seq.seq_id, next_token, token_text)

        if seq.is_finished():
            self.cache.delete(seq.seq_id)
            self.scheduler.mark_finished(seq)
        else:
            self.cache.allocate(seq.seq_id, seq.past_key_values, seq.current_position)

    def _prefill_batch(self, input_ids: torch.Tensor):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, None)
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def _decode_batch(self, input_ids: torch.Tensor, past_key_values: DynamicCache):
        inputs = self.adapter.prepare_inputs(self.model, input_ids, past_key_values)
        outputs = self.model(**inputs)
        return self.adapter.process_outputs(outputs, input_ids.size(1))

    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """采样下一个token"""
        logits = logits / temperature

        # Top-p 采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def stream_generate(self, prompt: str, max_tokens: int = 128,
                        temperature: float = 0.7, top_p: float = 0.9) -> Generator[Tuple[int, str], None, None]:
        """流式生成"""
        seq_id = self.add_request(prompt, max_tokens, temperature, top_p)
        token_queue = Queue()

        def callback(token, text):
            token_queue.put((token, text))

        self.register_stream_callback(seq_id, callback)

        try:
            generated_count = 0
            while generated_count < max_tokens:
                if self.step():
                    # 处理新生成的token
                    while not token_queue.empty():
                        token, text = token_queue.get()
                        yield token, text
                        generated_count += 1

                        if token == self.eos_token_id:
                            return

                # 检查序列状态
                if not any(seq.seq_id == seq_id for seq in self.scheduler.running_sequences):
                    break

                time.sleep(0.001)
        finally:
            self.unregister_stream_callback(seq_id)

    def generate(self, prompts: List[str], max_tokens: int = 100) -> Dict[Sequence, str]:
        """批量生成"""
        for prompt in prompts:
            self.add_request(prompt, max_tokens)

        # 执行推理直到完成
        for _ in range(max_tokens):
            if not self.step() and not self.scheduler.running_sequences:
                break

        # 处理结果
        results = {}
        finished_results = self.scheduler.get_finished_results()
        for seq, output_ids in finished_results:
            try:
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                results[seq] = text
            except Exception:
                results[seq] = f"[Error] {output_ids}"

        return results