import torch
from typing import Optional


class DynamicBuffer:
    """
    动态缓冲区类，用于管理KV缓存的内存分配
    """

    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.kv_key_buffer = None
        self.kv_value_buffer = None
        self.pos_buffer = None
        self.kv_buffer_size = 0  # 当前缓冲区的序列长度容量
        self.batch_buffer_size = 0  # 当前缓冲区的批次大小容量

    def _get_device(self, device_str: str) -> torch.device:
        """自动检测设备"""
        if device_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_str)

    def allocate_buffer(
            self,
            batch_size: int,
            kv_num_heads: int,
            head_size: int,
            max_seq_len: int,
            dtype: torch.dtype
    ) -> None:
        """
        分配缓冲区

        参数:
            batch_size: 批次大小
            kv_num_heads: KV头的数量
            head_size: 头维度大小
            max_seq_len: 最大序列长度
            dtype: 数据类型
        """
        # 计算新的缓冲区大小，适当增加余量
        new_batch_size = max(batch_size, self.batch_buffer_size)
        new_max_seq_len = max(max_seq_len, self.kv_buffer_size) * 2  # 动态增长，乘以2以减少频繁分配

        # 分配缓冲区
        self.kv_key_buffer = torch.zeros(
            new_batch_size, kv_num_heads, new_max_seq_len, head_size,
            device=self.device, dtype=dtype
        )
        self.kv_value_buffer = torch.zeros_like(self.kv_key_buffer)
        self.pos_buffer = torch.zeros(
            new_batch_size, new_max_seq_len,
            dtype=torch.long, device=self.device
        )

        self.batch_buffer_size = new_batch_size
        self.kv_buffer_size = new_max_seq_len

    def get_buffer_slice(
            self,
            batch_size: int,
            max_seq_len: int
    ) -> tuple:
        """
        获取缓冲区的切片

        参数:
            batch_size: 需要的批次大小
            max_seq_len: 需要的序列长度

        返回:
            (keys, values, positions) 的切片视图
        """
        if (self.kv_key_buffer is None or
                batch_size > self.batch_buffer_size or
                max_seq_len > self.kv_buffer_size):
            return None, None, None

        return (
            self.kv_key_buffer[:batch_size, :, :max_seq_len, :],
            self.kv_value_buffer[:batch_size, :, :max_seq_len, :],
            self.pos_buffer[:batch_size, :max_seq_len]
        )

    def check_and_allocate(
            self,
            batch_size: int,
            kv_num_heads: int,
            head_size: int,
            max_seq_len: int,
            dtype: torch.dtype
    ) -> bool:
        """
        检查缓冲区是否足够，不足则重新分配

        返回:
            True: 需要重新分配
            False: 缓冲区足够
        """
        if (self.kv_key_buffer is None or
                self.batch_buffer_size < batch_size or
                self.kv_buffer_size < max_seq_len):
            self.allocate_buffer(batch_size, kv_num_heads, head_size, max_seq_len, dtype)
            return True
        return False

    def clear(self) -> None:
        """清空缓冲区"""
        self.kv_key_buffer = None
        self.kv_value_buffer = None
        self.pos_buffer = None
        self.kv_buffer_size = 0
        self.batch_buffer_size = 0