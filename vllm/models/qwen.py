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
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

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
            torch_dtype=torch.float16,
            device_map="auto"  # +++ 添加自动设备映射 +++
        )

        # 应用张量并行
        if tensor_parallel_manager:
            instance.model = tensor_parallel_manager.apply_tensor_parallelism(instance.model)

        # 设置为评估模式
        instance.model.eval()

        return instance

    def __call__(self, input_ids, positions, past_key_values=None, use_cache=False):
        input_ids = input_ids.contiguous()
        positions = positions.contiguous()

        if past_key_values is not None:
            past_key_values = [(k.contiguous(), v.contiguous()) for k, v in past_key_values]

        attention_mask = self._prepare_attention_mask(input_ids, past_key_values)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        return outputs

    def _prepare_attention_mask(self, input_ids, past_key_values=None):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        num_heads = self.num_key_value_heads

        if past_key_values is None:
            causal_mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
            return causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, -1, -1)

        else:
            first_layer_k = past_key_values[0][0]
            past_len = first_layer_k.size(2)
            total_len = past_len + seq_length

            mask = torch.zeros(batch_size, num_heads, seq_length, total_len,
                               dtype=torch.bool, device=device)
            mask[:, :, :, :past_len] = 1

            causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
            mask[:, :, :, past_len:past_len + seq_length] = causal_mask.unsqueeze(0).unsqueeze(1)

            return mask