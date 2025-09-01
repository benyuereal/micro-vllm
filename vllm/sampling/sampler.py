import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 100
    stop_words: List[str] = None
    repetition_penalty: float = 1.0


class Sampler:
    """采样器，负责从logits中采样下一个token"""

    def __init__(self):
        pass

    def sample(self, logits: torch.Tensor, sampling_params: SamplingParams) -> torch.Tensor:
        """从logits中采样"""
        # 确保输入至少是2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)  # [batch_size, 1, vocab_size]

        # 现在logits总是三维
        batch_size, seq_len, vocab_size = logits.shape

        # 应用温度调节
        if sampling_params.temperature > 0:
            scaled_logits = logits / sampling_params.temperature
        else:
            scaled_logits = logits * 1e10  # 温度=0时使用贪婪采样

        # 应用top-k过滤
        if sampling_params.top_k > 0:
            self._apply_top_k(scaled_logits, sampling_params.top_k)

        # 应用top-p过滤
        if sampling_params.top_p < 1.0:
            self._apply_top_p(scaled_logits, sampling_params.top_p)

        # 计算概率分布
        probs = torch.softmax(scaled_logits, dim=-1)

        # 采样
        next_tokens = torch.multinomial(probs[:, -1, :], num_samples=1)
        return next_tokens.squeeze(-1)

    def _apply_repetition_penalty(self, logits: torch.Tensor, penalty: float):
        """应用重复惩罚"""
        # 简化实现：实际实现需要知道已生成的token
        pass

    def _apply_top_k(self, logits: torch.Tensor, k: int):
        """应用top-k过滤"""
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits[logits < min_values] = -float('inf')

    def _apply_top_p(self, logits: torch.Tensor, p: float):
        """应用top-p过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过p的token
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将需要移除的token的logits设为负无穷
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')