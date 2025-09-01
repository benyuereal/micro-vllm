# vllm/schema.py
from dataclasses import dataclass
from typing import Any, Optional
from .sampling.sampler import SamplingParams

@dataclass
class Request:
    request_id: int
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
    priority: int = 0

@dataclass
class Response:
    request_id: int
    generated_text: str
    success: bool
    error_message: Optional[str] = None