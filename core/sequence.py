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
        self.repetition_penalty = 1.0  # 1.0=禁用，>1.0 惩罚已出现 token（HF 约定）
        self.eos_token_id = tokenizer.eos_token_id
        self.priority = 0
        self.timestamp = time.time()  # 请求到达时间戳
        self.stop_strings = []        # 服务端停止字符串（命中即结束生成，避免 client 提前断流导致 seq 孤儿）
        # chunked prefill：已 prefill 的 token 数（KV 已写入 cache 的前缀长度）。
        # 0=尚未 prefill；==len(input_ids)=prefill 完成可转 decode。中间值=长 prompt 分块中。
        self.prefill_done = 0
        # 本 step 要 prefill 的 chunk 长度（由 scheduler 设置，engine _prefill 读取切片）。
        # 0 表示该 seq 本 step 不参与 prefill。
        self._chunk_len = 0
        # 本 chunk 是否是该 prompt 的最后一块（prefill_done + _chunk_len == len(input_ids)）。
        # 最后一块才采样首 token 并转 decode；中间块只写 KV、推进 prefill_done。
        self._chunk_is_last = True
        # is_finished 缓存：output_ids 只在 update_state 增长，故在那里算一次缓存，
        # 其余调用点（scheduler listcomp / update_sequences）直接读缓存避免重复调用。
        self._finished = False

    def is_finished(self):
        return self._finished

    @property
    def prefill_remaining(self) -> int:
        """尚未 prefill 的 prompt token 数。"""
        return len(self.input_ids) - self.prefill_done

    def get_next_input_ids(self):
        if self.state == "prefill":
            # chunked prefill：返回本 step 的 chunk（prefill_done 起的 _chunk_len 个 token）。
            # scheduler 已设好 _chunk_len；若为 0（未切），退化为返回剩余全部（原一次性 prefill）。
            end = self.prefill_done + (self._chunk_len or (len(self.input_ids) - self.prefill_done))
            return self.input_ids[self.prefill_done:end]
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

        # 缓存 is_finished（output_ids 仅此处增长）
        self._finished = (len(self.output_ids) >= self.max_tokens or
                          next_token == self.eos_token_id)
        if self._finished:
            self.state = "finished"
        elif self.state == "prefill":
            self.state = "decode"

    def advance_prefill(self, chunk_len: int):
        """chunked prefill 完成一个 chunk 后推进 prefill_done。

        - 不产生新 token，仅记录已写入 cache 的前缀长度。
        - prefill_done 达到 prompt 长度时，prefill 完成：本方法返回后由 scheduler
          在下一步将 seq 推入 decode（prefill 完成的最后一步会由 prefill_runner
          产生首 token 并调用 update_state 转 decode，所以这里只推进中间 chunk）。
        """
        self.prefill_done = min(self.prefill_done + chunk_len, len(self.input_ids))
        # current_position 反映已处理位置；chunked 期间它等于 prefill_done（KV 已写到此处）
        self.current_position = self.prefill_done

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
            "prefill_done": self.prefill_done,
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
        seq.prefill_done = data.get("prefill_done", 0)
        seq.temperature = data["temperature"]
        seq.top_p = data["top_p"]
        seq.eos_token_id = data["eos_token_id"]
        seq.priority = data["priority"]
        seq.timestamp = data["timestamp"]
        seq._next_token = data["_next_token"]
        seq._finished = (seq.state == "finished")

        # 非主Rank不需要past_key_values，保持None即可，不影响推理
        return seq
