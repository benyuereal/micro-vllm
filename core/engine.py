# core/engine.py (重构)
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from . import Scheduler
from .cache_manager import KVCache
from .model_loader import load_model
from .sequence import Sequence
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size=8):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.cache = KVCache(max_batch_size)
        self.adapter = QwenModelAdapter()
        self.device = next(self.model.parameters()).device
        self.max_batch_size = max_batch_size

    def batch_prefill(self, sequences: List[Sequence]) -> Dict[int, torch.Tensor]:
        """批量处理初始提示"""
        batch_inputs = []
        batch_sequences = []

        # 准备批处理输入
        for seq in sequences:
            input_ids = self.tokenizer.encode(seq.prompt, return_tensors="pt").to(self.device)
            seq.input_ids = input_ids[0].tolist()
            batch_inputs.append(input_ids)
            batch_sequences.append(seq)

        # 创建注意力掩码
        input_lengths = [tensor.size(1) for tensor in batch_inputs]
        max_length = max(input_lengths)

        padded_inputs = torch.zeros(len(batch_inputs), max_length, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(len(batch_inputs), max_length, dtype=torch.long, device=self.device)

        for i, tensor in enumerate(batch_inputs):
            padded_inputs[i, :input_lengths[i]] = tensor[0]
            attention_mask[i, :input_lengths[i]] = 1

        # 准备模型输入
        inputs = self.adapter.prepare_batch_inputs(
            self.model,
            padded_inputs,
            attention_mask,
            None
        )

        # 执行推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 处理输出并分配缓存
        logits, past_key_values = self.adapter.process_batch_outputs(outputs)

        results = {}
        for i, seq in enumerate(batch_sequences):
            # 直接取最后一个token的logits
            last_token_idx = input_lengths[i] - 1
            seq_logits = logits[i, last_token_idx, :]  # 形状 (vocab_size)

            # 存储为2维张量
            results[seq.seq_id] = seq_logits.unsqueeze(0)  # (1, vocab_size)

            self.cache.allocate(seq.seq_id, tuple(pk[i:i + 1] for pk in past_key_values))
            seq.prefill_done = True

        return results

    def batch_decode(self, sequences: List[Sequence]) -> Dict[int, torch.Tensor]:
        """批量处理解码步骤"""
        if not sequences:
            return {}

        # 准备输入tokens
        input_tokens = torch.tensor(
            [seq.get_last_token() for seq in sequences],
            device=self.device
        ).unsqueeze(1)

        # 获取批处理缓存
        seq_ids = [seq.seq_id for seq in sequences]
        past_key_values = self.cache.get_batch(seq_ids)

        # 准备模型输入
        inputs = self.adapter.prepare_batch_inputs(
            self.model,
            input_tokens,
            None,
            past_key_values
        )

        # 执行推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 处理输出并更新缓存
        logits, new_past_key_values = self.adapter.process_batch_outputs(outputs)

        # 更新缓存
        for i, seq_id in enumerate(seq_ids):
            self.cache.update(seq_id, tuple(pk[i:i + 1] for pk in new_past_key_values))

        # 返回每个序列的logits
        return {seq.seq_id: logits[i:i + 1, -1, :] for i, seq in enumerate(sequences)}

    # core/engine.py (修改部分)
    # core/engine.py (修改 sample_next_tokens 方法)

    def sample_next_tokens(self, logits_dict: Dict[int, torch.Tensor],
                           temperature: float = 0.7, top_p: float = 0.9) -> Dict[int, int]:
        """批量采样下一个token"""
        next_tokens = {}

        for seq_id, logits in logits_dict.items():
            # 确保logits是2维：[batch_size, vocab_size]
            original_shape = logits.shape
            if logits.dim() == 3:
                # 如果是预填充阶段返回的logits (batch_size, seq_len, vocab_size)
                # 只取最后一个token的logits
                logits = logits[:, -1, :]  # 形状变为 (batch_size, vocab_size)

            # 现在logits应该是2维: (1, vocab_size) 或 (N, vocab_size)
            if logits.dim() != 2:
                # 如果仍然不是2维，调整为2维
                logits = logits.view(1, -1)

            # 应用温度
            logits = logits / temperature

            # 应用top-p（核采样）
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率低于top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 关键修复：确保indices_to_remove与logits维度匹配
            # 获取需要移除的索引
            removal_indices = sorted_indices[sorted_indices_to_remove]

            # 创建与logits形状相同的索引掩码
            indices_to_remove = torch.full_like(logits, fill_value=-1, dtype=torch.long)
            indices_to_remove[sorted_indices_to_remove] = removal_indices

            # 安全移除
            logits = logits.scatter(-1, indices_to_remove, float('-inf'))

            # 从剩余token中采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_tokens[seq_id] = next_token

        return next_tokens

    def generate(self, prompts: List[str], max_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9) -> Dict[int, str]:
        """支持连续批处理的生成方法"""
        scheduler = Scheduler(max_batch_size=self.max_batch_size)
        results = {}

        # 添加初始请求
        for prompt in prompts:
            seq_id = scheduler.add_request(prompt)
            results[seq_id] = []

        # 主处理循环
        for _ in range(max_tokens):
            # 步骤1: 处理新的预填充请求
            prefill_batch = scheduler.get_prefill_batch()
            if prefill_batch:
                prefill_logits = self.batch_prefill(prefill_batch)
                next_tokens = self.sample_next_tokens(prefill_logits, temperature, top_p)

                for seq in prefill_batch:
                    token = next_tokens.get(seq.seq_id, self.tokenizer.eos_token_id)
                    seq.add_output_token(token)
                    results[seq.seq_id].append(token)

                scheduler.move_to_running(prefill_batch)

            # 步骤2: 处理正在运行的序列
            decode_batch = scheduler.get_decode_batch()
            if decode_batch:
                decode_logits = self.batch_decode(decode_batch)
                next_tokens = self.sample_next_tokens(decode_logits, temperature, top_p)

                for seq in decode_batch:
                    token = next_tokens.get(seq.seq_id, self.tokenizer.eos_token_id)

                    # 检查结束条件
                    if token == self.tokenizer.eos_token_id:
                        scheduler.mark_finished(seq.seq_id)
                    else:
                        seq.add_output_token(token)
                        results[seq.seq_id].append(token)

            # 检查是否所有序列都已完成
            if not scheduler.has_requests():
                break

        # 释放缓存并转换结果
        self.cache.free_inactive()
        decoded_results = {}
        for seq_id, token_ids in results.items():
            try:
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                decoded_results[seq_id] = decoded_text
            except Exception as e:
                print(f"Error decoding sequence {seq_id}: {str(e)}")
                decoded_results[seq_id] = f"[Decoding error]"

        return decoded_results