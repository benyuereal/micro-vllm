from dataclasses import dataclass, field
from typing import List
import torch
import torch.distributed as dist
from core.parallel_config import get_rank, rank0
from core.sequence import Sequence

@dataclass
class BatchInferenceContext:
    # 批次基础信息
    batch_size: int
    batch_type: str  # prefill / decode / waiting
    # 直接携带Sequence对象列表
    sequences: List[Sequence] = field(default_factory=list)
    # 设备缓存
    _device: torch.device = None

    def __post_init__(self):
        # 你可以改成自己的获取当前rank设备的方法
        self._device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 核心方法：主Rank广播上下文（仅传输Sequence的必要字段）
    # ------------------------------
    def broadcast(self):
        if not rank0():
            raise RuntimeError("broadcast can only be called on main rank (Rank 0)")
        
        device = self._device

        # 1. 先广播批次基础信息
        base_meta = torch.tensor([
            self.batch_size,
            0 if self.batch_type == "waiting" else 1 if self.batch_type == "prefill" else 2,
        ], dtype=torch.long, device=device)
        dist.broadcast(base_meta, src=0)

        if self.batch_type == "waiting":
            return

        # 2. 广播Sequence的必要字段（过滤掉tokenizer、prompt等大字段）
        seq_meta_list = []
        max_input_len = max(len(seq.get_next_input_ids()) for seq in self.sequences)
        input_ids_padded = []

        for seq in self.sequences:
            input_ids = seq.get_next_input_ids()
            # 仅传输推理必需的元信息
            seq_meta = [
                seq.seq_id,
                seq.current_position,
                len(input_ids),  # 本次推理的输入长度
                seq.temperature,
                seq.top_p,
                seq.max_tokens,
                1 if seq.is_finished() else 0,
                seq.state == "prefill",  # 标记是否是prefill状态
                seq._next_token if seq._next_token is not None else -1  # 采样结果
            ]
            seq_meta_list.append(seq_meta)
            # 输入id padding
            input_ids_padded.append(input_ids + [0] * (max_input_len - len(input_ids)))

        # 转tensor广播
        seq_meta_tensor = torch.tensor(seq_meta_list, dtype=torch.float32, device=device)
        input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
        max_input_len_tensor = torch.tensor([max_input_len], dtype=torch.long, device=device)

        dist.broadcast(max_input_len_tensor, src=0)
        dist.broadcast(seq_meta_tensor, src=0)
        dist.broadcast(input_ids_tensor, src=0)

    # ------------------------------
    # 核心方法：非主Rank接收并还原轻量Sequence对象
    # ------------------------------
    @classmethod
    def receive(cls, dummy_tokenizer):
        """
        dummy_tokenizer: 非主Rank需要传入一个空的tokenizer（仅用于初始化Sequence，不会实际使用）
        """
        if rank0():
            raise RuntimeError("receive can only be called on non-main rank")
        
        device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")

        # 1. 接收批次基础信息
        base_meta = torch.empty(2, dtype=torch.long, device=device)
        dist.broadcast(base_meta, src=0)
        batch_size = base_meta[0].item()
        batch_type_code = base_meta[1].item()

        batch_type_map = {0: "waiting", 1: "prefill", 2: "decode"}
        batch_type = batch_type_map[batch_type_code]

        ctx = cls(batch_size=batch_size, batch_type=batch_type, _device=device)
        if batch_type == "waiting":
            return ctx

        # 2. 接收Sequence元信息
        max_input_len_tensor = torch.empty(1, dtype=torch.long, device=device)
        dist.broadcast(max_input_len_tensor, src=0)
        max_input_len = max_input_len_tensor[0].item()

        seq_meta_tensor = torch.empty((batch_size, 9), dtype=torch.float32, device=device)
        input_ids_tensor = torch.empty((batch_size, max_input_len), dtype=torch.long, device=device)

        dist.broadcast(seq_meta_tensor, src=0)
        dist.broadcast(input_ids_tensor, src=0)

        # 3. 还原轻量Sequence对象（非主Rank用）
        sequences = []
        for i in range(batch_size):
            meta = seq_meta_tensor[i].tolist()
            seq_id = int(meta[0])
            current_position = int(meta[1])
            input_len = int(meta[2])
            temperature = meta[3]
            top_p = meta[4]
            max_tokens = int(meta[5])
            is_finished = bool(meta[6])
            is_prefill = bool(meta[7])
            next_token = int(meta[8]) if meta[8] != -1 else None

            # 还原input_ids
            input_ids = input_ids_tensor[i][:input_len].tolist()

            # 创建轻量Sequence（用空prompt和dummy_tokenizer初始化，仅用于携带数据）
            seq = Sequence(seq_id=seq_id, prompt="", tokenizer=dummy_tokenizer, max_tokens=max_tokens)
            # 手动覆盖必要字段
            seq.temperature = temperature
            seq.top_p = top_p
            seq.current_position = current_position
            seq.state = "prefill" if is_prefill else "decode"
            seq._next_token = next_token
            seq.output_ids = []  # 非主Rank不需要真实output_ids
            seq.full_ids = input_ids[:]
            # 手动设置input_ids，确保get_next_input_ids()返回正确值
            if is_prefill:
                seq.input_ids = input_ids
            else:
                seq.input_ids = []
                seq.output_ids = input_ids  # decode阶段get_next_input_ids()取output_ids[-1]
            
            sequences.append(seq)

        ctx.sequences = sequences
        return ctx

