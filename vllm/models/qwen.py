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

    # 在文档3的 QwenModel 类中修改 __call__ 方法
    def __call__(self,
                 input_ids: torch.Tensor,
                 positions: torch.Tensor,
                 past_key_values: Optional[Tuple] = None,
                 use_cache: bool = False):
        # 确保输入连续
        input_ids = input_ids.contiguous()
        positions = positions.contiguous()

        # 准备注意力掩码
        attention_mask = self._prepare_attention_mask(input_ids, past_key_values)

        # 执行模型前向传播 - 关键修复：添加 output_hidden_states=True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True  # 确保返回隐藏状态
        )

        # 返回完整输出对象
        return outputs

    # qwen.py 中的 _prepare_attention_mask 方法
    def _prepare_attention_mask(self,
                                input_ids: torch.Tensor,
                                past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """准备符合Qwen要求的注意力掩码"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if past_key_values is None:
            print("No past key values - creating 1D attention mask")
            # 创建一维序列长度指示器 [batch_size, seq_length]
            attention_mask = torch.ones(
                batch_size, seq_length,
                dtype=torch.long, device=device
            )
            return attention_mask.contiguous()

        print("Using past key values - creating extended mask")
        # 获取历史长度
        first_layer_k = past_key_values[0][0]
        past_lengths = first_layer_k.size(2)  # [batch, heads, seq_len, head_dim]
        total_length = past_lengths + seq_length

        # 创建扩展掩码 [batch_size, total_length]
        attention_mask = torch.cat([
            torch.ones(batch_size, past_lengths, device=device),
            torch.ones(batch_size, seq_length, device=device)
        ], dim=1)

        print(f"Extended mask shape: {attention_mask.shape}")
        return attention_mask.contiguous()