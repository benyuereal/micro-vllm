# vllm/schema.py
from dataclasses import dataclass, field
from typing import Any, Optional, List

import torch

from .sampling.sampler import SamplingParams

@dataclass
class Request:
    request_id: int
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
    priority: int = 0
    # 添加以下字段确保正确初始化
    generated_token_ids: List[int] = field(default_factory=list)
    prompt_ids: torch.Tensor = None
    remaining_tokens: int = 0
    is_completed: bool = False
    start_time: float = 0.0
    last_decoded_index: int = 0  # 添加最后解码位置属性

@dataclass
class Response:
    request_id: int
    generated_text: str
    success: bool
    error_message: Optional[str] = None