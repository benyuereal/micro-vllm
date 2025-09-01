# vllm/schema.py
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple

import torch
from torch import Tensor

from .sampling.sampler import SamplingParams

@dataclass
class Request:
    request_id: int
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float  # 请求到达时间
    priority: int = 0  # 优先级（默认0）

    # 状态字段（由LLM引擎填充）
    generated_token_ids: List[int] = field(default_factory=list)  # 已生成的token ID列表
    prompt_ids: Optional[torch.Tensor] = None  # 编码后的提示token（张量）
    remaining_tokens: int = 0  # 剩余可生成token数（初始为sampling_params.max_tokens）
    is_completed: bool = False  # 请求是否完成
    start_time: float = 0.0  # 请求开始处理时间
    last_decoded_index: int = 0  # 最后解码的token索引（用于增量生成）

    def __post_init__(self):
        """确保所有必要属性正确初始化"""
        # 确保有生成的token列表
        if not hasattr(self, 'generated_token_ids'):
            self.generated_token_ids = []

        # 确保剩余token数正确
        if self.remaining_tokens <= 0:
            self.remaining_tokens = self.sampling_params.max_tokens

        # 确保开始时间有效
        if self.start_time <= 0:
            self.start_time = time.time()

@dataclass
class Response:
    """LLM响应数据结构"""
    request_id: int
    generated_text: str  # 生成的文本
    success: bool  # 是否成功
    error_message: Optional[str] = None  # 错误信息（如果有）
    metadata: Optional[Dict[str, Any]] = None  # 附加元数据

@dataclass
class GenerationOutput:
    """模型生成输出"""
    sequences: List[List[int]]  # 生成的token序列
    scores: Optional[List[List[float]]] = None  # 每个token的生成概率

@dataclass
class ModelOutput:
    """模型前向传播输出"""
    logits: Tensor  # 预测的logits
    hidden_states: Optional[Tensor] = None  # 最后一层隐藏状态
    past_key_values: Optional[Tuple] = None  # 过去的关键值缓存
    attentions: Optional[Tuple] = None  # 注意力权重