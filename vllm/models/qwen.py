import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import AutoModelForCausalLM
from typing import Optional, Tuple
from .model_registry import register_model



@register_model("Qwen")
class QwenModel:
    """Qwen模型实现"""

    def __init__(self, config):
        self.config = config
        self.model = None

    @classmethod
    def initialize(cls,
                   model_name: str,
                   tensor_parallel_manager=None,
                   memory_manager=None):
        """初始化模型"""
        # 加载配置
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # 创建模型实例
        instance = cls(config)

        # 加载模型权重
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # 应用张量并行
        if tensor_parallel_manager:
            instance.model = tensor_parallel_manager.apply_tensor_parallelism(instance.model)

        # 设置为评估模式
        instance.model.eval()

        return instance

    def __call__(self,
                 input_ids: torch.Tensor,
                 positions: torch.Tensor,
                 past_key_values: Optional[Tuple] = None,
                 use_cache: bool = False):
        """前向传播"""
        # 准备注意力掩码
        attention_mask = self._prepare_attention_mask(input_ids, past_key_values)

        # 执行模型前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        return outputs

    def _prepare_attention_mask(self,
                                input_ids: torch.Tensor,
                                past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """准备注意力掩码（兼容分页缓存结构）"""
        batch_size, seq_length = input_ids.shape

        # 如果past_key_values存在，提取实际长度
        if past_key_values is not None:
            # 获取第一个序列在第一个层的k缓存长度
            if past_key_values and len(past_key_values) > 0:
                # 取第0层的k缓存 [batch, num_heads, seq_len, head_size]
                k_batch = past_key_values[0][0]
                # 所有序列共享相同的缓存长度
                past_length = k_batch.size(2)  # 取seq_len维度
                past_lengths = [past_length] * batch_size
            else:
                past_lengths = [0] * batch_size
        else:
            past_lengths = [0] * batch_size

        # 创建当前token的因果掩码
        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        # 无历史缓存时直接返回
        if max(past_lengths) == 0:
            return causal_mask.expand(batch_size, 1, seq_length, seq_length)

        # 创建扩展注意力掩码
        max_past_len = max(past_lengths)
        total_length = max_past_len + seq_length
        extended_mask = torch.zeros(
            batch_size, 1, seq_length, total_length,
            dtype=torch.bool, device=input_ids.device
        )

        # 填充掩码：
        # - 历史部分：全部可见（True）
        # - 当前部分：因果可见（左下三角）
        for i, past_len in enumerate(past_lengths):
            # 历史部分
            if past_len > 0:
                extended_mask[i, :, :, :past_len] = True

            # 当前部分（因果）
            extended_mask[i, :, :, past_len:past_len + seq_length] = causal_mask

        return extended_mask