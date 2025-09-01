# core/engine.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .cache_manager import KVCache
from .model_loader import load_model
from .batch_manager import BatchManager
from models.qwen_adapter import QwenModelAdapter


class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size: int = 8):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()
        self.cache = KVCache()
        self.adapter = QwenModelAdapter
        self.device = next(self.model.parameters()).device
        self.batch_manager = BatchManager(max_batch_size)
        self.max_batch_size = max_batch_size

    def prefill(self, batch: List[Tuple[int, torch.Tensor]]) -> Tuple[Dict[int, torch.Tensor], Dict[int, Tuple]]:
        """批量处理初始提示（prefill阶段）"""
        # 按序列长度排序以提高效率
        batch.sort(key=lambda x: x[1].size(1), reverse=True)

        seq_ids = [item[0] for item in batch]
        input_tensors = [item[1] for item in batch]

        # 拼接输入并创建注意力掩码
        input_ids = torch.cat(input_tensors, dim=0)
        attention_mask = torch.ones_like(input_ids)

        # 准备模型输入
        inputs = self.adapter.prepare_batch_inputs(
            self.model,
            input_ids,
            attention_mask,
            None
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 处理输出并分割回各个序列
        logits, past_key_values = self.adapter.process_batch_outputs(
            outputs,
            [tensor.size(1) for tensor in input_tensors]
        )

        # 创建结果字典
        result_logits = {}
        result_kv = {}

        for i, seq_id in enumerate(seq_ids):
            result_logits[seq_id] = logits[i]
            result_kv[seq_id] = past_key_values[i]

        return result_logits, result_kv

    def decode(self, batch: List[Tuple[int, torch.Tensor]]) -> Tuple[Dict[int, torch.Tensor], Dict[int, Tuple]]:
        """批量解码（decode阶段）"""
        seq_ids = [item[0] for item in batch]
        input_tensors = [item[1] for item in batch]

        # 获取各个序列的past_key_values
        past_key_values_list = []
        for seq_id in seq_ids:
            cache_entry = self.cache.get(seq_id)
            if cache_entry and cache_entry['past_key_values'] is not None:
                past_key_values_list.append(cache_entry['past_key_values'])
            else:
                # 如果没有缓存，创建一个空的（这种情况不应该发生）
                past_key_values_list.append(None)

        # 拼接输入
        input_ids = torch.cat(input_tensors, dim=0)

        # 准备模型输入
        inputs = self.adapter.prepare_batch_inputs(
            self.model,
            input_ids,
            None,  # decode阶段不需要注意力掩码
            past_key_values_list
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 处理输出
        logits, past_key_values = self.adapter.process_batch_outputs(
            outputs,
            [1] * len(batch)  # 每个序列只输入了一个token
        )

        # 创建结果字典
        result_logits = {}
        result_kv = {}

        for i, seq_id in enumerate(seq_ids):
            result_logits[seq_id] = logits[i]
            result_kv[seq_id] = past_key_values[i]

        return result_logits, result_kv

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
        """生成文本 - 支持连续批处理"""
        # 初始化序列
        for prompt in prompts:
            seq_id = id(prompt)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            self.batch_manager.add_sequence(seq_id, input_ids)
            self.cache.allocate(seq_id)

        results = {}
        active_sequences = set(id(prompt) for prompt in prompts)

        # 处理阶段：prefill + decode
        for step in range(max_tokens):
            # 获取当前批次
            batch = self.batch_manager.get_batch()
            if not batch:
                break

            if step == 0:
                # Prefill阶段 - 处理初始提示
                logits_dict, kv_dict = self.prefill(batch)
            else:
                # Decode阶段 - 处理单个token
                logits_dict, kv_dict = self.decode(batch)

            # 更新缓存和处理结果
            for seq_id in logits_dict:
                # 更新缓存
                self.cache.update(seq_id, kv_dict[seq_id])

                # 采样下一个token
                next_token = self.sample_next_token(
                    logits_dict[seq_id][0, -1, :],
                    temperature,
                    top_p
                )

                # 检查是否结束
                if next_token == self.tokenizer.eos_token_id:
                    self.batch_manager.finish_sequence(seq_id)
                    active_sequences.discard(seq_id)
                    continue

                # 添加到结果中
                if seq_id not in results:
                    results[seq_id] = self.batch_manager.sequences[seq_id]['tokens']
                results[seq_id].append(next_token)

                # 更新批次管理器
                self.batch_manager.update_sequence(
                    seq_id,
                    torch.tensor([[next_token]], device=self.device)
                )

            # 如果没有活跃序列，提前结束
            if not active_sequences:
                break

        # 清理缓存
        for seq_id in list(self.cache.cache.keys()):
            if seq_id not in results:
                self.cache.remove(seq_id)

        # 结果转换
        decoded_results = {}
        for seq_id, token_ids in results.items():
            try:
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                # 找到对应的prompt
                prompt = next(p for p in prompts if id(p) == seq_id)
                # 移除可能的重复内容
                if decoded_text.startswith(prompt):
                    decoded_text = decoded_text[len(prompt):]
                decoded_results[seq_id] = decoded_text
            except Exception as e:
                print(f"Error decoding sequence {seq_id}: {str(e)}")
                decoded_results[seq_id] = f"[Decoding error] Token IDs: {token_ids}"

        return decoded_results