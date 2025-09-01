from dataclasses import dataclass
from typing import List, Optional
import uuid
import time


@dataclass
class Request:
    """推理请求"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 0.9

    # 内部状态
    output_tokens: List[int] = None
    remaining_tokens: int = None
    finished: bool = False
    block_ids: List[int] = None
    created_time: float = None

    def __post_init__(self):
        if self.output_tokens is None:
            self.output_tokens = []
        if self.remaining_tokens is None:
            self.remaining_tokens = self.max_tokens
        if self.created_time is None:
            self.created_time = time.time()

    @classmethod
    def from_prompt(cls, prompt: str, max_tokens: int = 100, **kwargs):
        """从提示创建请求"""
        return cls(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs
        )