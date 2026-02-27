from typing import Tuple, Union, List, Optional
import torch
import time


class Sequence:
    def __init__(self, seq_id: int, prompt: str, tokenizer, max_tokens: int = 128):
        self.seq_id = seq_id
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        self.output_ids = []
        self.full_ids = self.input_ids[:]
        self.state = "prefill"  # prefill / decode / finished
        self.past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.current_position = len(self.input_ids)
        self.temperature = 0.7
        self.top_p = 0.9
        self.eos_token_id = tokenizer.eos_token_id
        self.priority = 0
        self.timestamp = time.time()  # 请求到达时间戳

    def is_finished(self):
        return (len(self.output_ids) >= self.max_tokens or
                (self.output_ids and self.output_ids[-1] == self.eos_token_id))

    def get_next_input_ids(self):
        if self.state == "prefill":
            return self.input_ids
        elif self.state == "decode":
            return [self.output_ids[-1]]
        return None

    # 用于非主Rank存储推理结果（临时属性，不持久化）
    _next_token: int = None

    def update_state(self, next_token: int, new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.output_ids.append(next_token)
        self.full_ids.append(next_token)
        self.past_key_values = new_past_key_values
        self.current_position += 1

        if self.is_finished():
            self.state = "finished"
        elif self.state == "prefill":
            self.state = "decode"

    def to_dict(self) -> dict:
        """
        提取推理必需的轻量字段，自动过滤不可序列化/大对象
        对所有列表类型显式复制，避免引用共享
        仅保留基础类型，保证分布式传输、JSON序列化无异常
        """
        return {
            # 核心标识字段
            "seq_id": self.seq_id,
            "max_tokens": self.max_tokens,
            "input_ids": self.input_ids[:],
            "output_ids": self.output_ids[:],
            "full_ids": self.full_ids[:],
            # 状态控制字段
            "state": self.state,
            "current_position": self.current_position,
            # 采样参数
            "temperature": self.temperature,
            "top_p": self.top_p,
            # 终止控制
            "eos_token_id": self.eos_token_id,
            # 调度相关
            "priority": self.priority,
            "timestamp": self.timestamp,
            # 临时推理结果
            "_next_token": self._next_token,
        }

    @classmethod
    def from_dict(cls, data: dict, dummy_tokenizer):
        """
        从字典还原Sequence实例（非主Rank专用）
        :param data: to_dict()输出的字典
        :param dummy_tokenizer: 空分词器，仅用于初始化，不会实际使用
        :return: 还原后的Sequence实例
        """
        # 用空prompt初始化基础实例，避免重新分词
        seq = cls(
            seq_id=data["seq_id"],
            prompt="",
            tokenizer=dummy_tokenizer,
            max_tokens=data["max_tokens"]
        )

        # 批量覆盖核心字段（✅ 显式切片复制，避免引用共享）
        seq.input_ids = data["input_ids"][:]
        seq.output_ids = data["output_ids"][:]
        seq.full_ids = data["full_ids"][:]
        
        # 覆盖其他状态字段
        seq.state = data["state"]
        seq.current_position = data["current_position"]
        seq.temperature = data["temperature"]
        seq.top_p = data["top_p"]
        seq.eos_token_id = data["eos_token_id"]
        seq.priority = data["priority"]
        seq.timestamp = data["timestamp"]
        seq._next_token = data["_next_token"]

        # 非主Rank不需要past_key_values，保持None即可，不影响推理
        return seq
