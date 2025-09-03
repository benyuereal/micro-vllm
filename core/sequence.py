# core/sequence.py
from typing import Tuple, Union

from transformers import DynamicCache


class Sequence:
    def __init__(self, seq_id: int, prompt: str, tokenizer, max_tokens: int = 128):
        self.seq_id = seq_id
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        self.output_ids = []
        self.full_ids = self.input_ids[:]  # 用于拼接输入
        self.state = "prefill"  # prefill / decode / finished
        self.past_key_values = None
        self.current_position = len(self.input_ids)  # 当前解码位置
        self.temperature = 0.7
        self.top_p = 0.9
        self.eos_token_id = tokenizer.eos_token_id
        self.priority = 0  # 用于抢占

    def is_finished(self):
        return (len(self.output_ids) >= self.max_tokens or
                (self.output_ids and self.output_ids[-1] == self.eos_token_id))

    def get_next_input_ids(self):
        if self.state == "prefill":
            return self.input_ids
        elif self.state == "decode":
            return [self.output_ids[-1]]  # 单个 token
        return None

    def update_state(self, next_token: int, new_past_key_values: Union[Tuple, DynamicCache]):
        self.output_ids.append(next_token)
        self.full_ids.append(next_token)

        # 统一转换为DynamicCache格式
        if isinstance(new_past_key_values, tuple):
            self.past_key_values = DynamicCache.from_legacy_cache(new_past_key_values)
        else:
            self.past_key_values = new_past_key_values

        self.current_position += 1
        if self.is_finished():
            self.state = "finished"
        elif self.state == "prefill":
            self.state = "decode"

