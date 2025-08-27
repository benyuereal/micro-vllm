import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
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
        instance.model = AutoModel.from_pretrained(
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
        """准备注意力掩码"""
        batch_size, seq_length = input_ids.shape

        # 创建因果注意力掩码
        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)

        # 如果有历史缓存，扩展掩码
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2) if past_key_values[0][0].nelement() > 0 else 0
            total_length = past_length + seq_length

            # 创建扩展的掩码
            extended_mask = torch.ones(
                batch_size, 1, seq_length, total_length,
                dtype=torch.bool, device=input_ids.device
            )

            # 应用因果掩码
            extended_mask[:, :, :, :past_length] = True  # 历史部分全部可见
            extended_mask[:, :, :, past_length:] = causal_mask

            return extended_mask

        return causal_mask.expand(batch_size, 1, seq_length, seq_length)