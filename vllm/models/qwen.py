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
        # 添加键值头数量配置
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

    def __call__(self,
                 input_ids: torch.Tensor,
                 positions: torch.Tensor,
                 past_key_values: Optional[Tuple] = None,
                 use_cache: bool = False):
        """前向传播"""
        # 添加：
        input_ids = input_ids.contiguous()
        positions = positions.contiguous()

        if past_key_values is not None:
            past_key_values = [
                (k.contiguous(), v.contiguous()) for k, v in past_key_values
            ]
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

    # qwen.py 中的 _prepare_attention_mask 方法
    def _prepare_attention_mask(self,
                                input_ids: torch.Tensor,
                                past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """准备注意力掩码（兼容Qwen模型要求）"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        print(f"Preparing attention mask for seq_length={seq_length}")

        if past_key_values is None:
            print("No past key values - creating causal mask")
            # Qwen模型在首次生成时需要特殊的4D因果掩码
            # 创建 [seq_length, seq_length] 的因果掩码
            causal_mask = torch.tril(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
            )

            # 扩展为 [batch_size, 1, seq_length, seq_length]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度
            causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

            print(f"Causal mask shape: {causal_mask.shape}")
            return causal_mask.contiguous()

        print("Using past key values - creating extended mask")
        # 有历史缓存时的处理
        first_layer_k = past_key_values[0][0]
        past_lengths = first_layer_k.size(2)  # [batch, heads, seq_len, head_dim]
        total_length = past_lengths + seq_length

        # 创建扩展的注意力掩码 [batch_size, 1, seq_length, total_length]
        extended_attention_mask = torch.zeros(
            batch_size,
            1,  # 单头掩码，模型会广播到多头
            seq_length,
            total_length,
            dtype=torch.bool,
            device=device
        )

        # 历史部分全部可见
        extended_attention_mask[:, :, :, :past_lengths] = 1

        # 当前部分：因果掩码（只允许关注自身及之前的token）
        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

        # 将因果掩码应用到当前部分
        extended_attention_mask[:, :, :, past_lengths:past_lengths + seq_length] = causal_mask

        print(f"Extended mask shape: {extended_attention_mask.shape}")
        return extended_attention_mask.contiguous()