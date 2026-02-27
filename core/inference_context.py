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
    batch_type: str  # 直接用字符串："waiting" / "prefill" / "decode"
    # 直接携带Sequence对象列表
    sequences: List[Sequence] = field(default_factory=list)
    # 设备缓存
    _device: torch.device = None

    def __post_init__(self):
        self._device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 主Rank广播：所有元数据统一用broadcast_object_list
    # ------------------------------
    def broadcast(self):
        if not rank0():
            raise RuntimeError("broadcast can only be called on main rank (Rank 0)")
        
        # 1. 打包所有基础元信息，一次性广播（变量名已优化）
        batch_size_meta = {
            "batch_size": self.batch_size,
            "batch_type": self.batch_type
        }
        dist.broadcast_object_list([batch_size_meta], src=0)

        # waiting类型无额外数据，直接结束
        if self.batch_type == "waiting":
            return

        # 2. 广播Sequence序列化后的字典列表
        seq_dict_list = [seq.to_dict() for seq in self.sequences]
        dist.broadcast_object_list(seq_dict_list, src=0)

    # ------------------------------
    # 非主Rank接收：和broadcast一一对应，成对调用
    # ------------------------------
    @classmethod
    def receive(cls, _tokenizer):
        """
        非主Rank专用：和主Rank的broadcast()必须成对调用，否则会卡死
        dummy_tokenizer: 空分词器，仅用于初始化Sequence
        """
        if rank0():
            raise RuntimeError("receive can only be called on non-main rank")
        
        # 1. 接收基础元信息（变量名已优化）
        batch_size_meta_container = [None]
        dist.broadcast_object_list(batch_size_meta_container, src=0)
        batch_size_meta = batch_size_meta_container[0]
        batch_size = batch_size_meta["batch_size"]
        batch_type = batch_size_meta["batch_type"]

        # 初始化上下文
        ctx = cls(batch_size=batch_size, batch_type=batch_type)
        if batch_type == "waiting":
            return ctx

        # 2. 接收Sequence字典列表，批量还原
        seq_dict_list = [None] * batch_size
        dist.broadcast_object_list(seq_dict_list, src=0)
        ctx.sequences = [Sequence.from_dict(seq_dict, _tokenizer) for seq_dict in seq_dict_list]

        return ctx