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

    def _prepare_attention_mask(self,
                                input_ids: torch.Tensor,
                                past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """准备注意力掩码（兼容分页缓存结构和分组查询注意力）"""
        batch_size, seq_length = input_ids.shape

        # 如果没有历史缓存，只需创建因果掩码
        if past_key_values is None:
            # 创建因果掩码 [batch_size, 1, target_length, target_length]
            attention_mask = torch.tril(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

            # 扩展掩码到键值头数（使用配置中的值）
            attention_mask = attention_mask.expand(batch_size, self.num_key_value_heads, seq_length, seq_length)
            return attention_mask

        # 获取每个序列的实际历史长度
        # 注意：past_key_values 的结构是：
        # 元组(每层一个元组(每层包含k_cache, v_cache))
        # k_cache 形状为 [batch_size, num_key_value_heads, past_length, head_size]
        # 我们使用第一个序列的k缓存来确定历史长度
        first_layer_k = past_key_values[0][0]  # 获取第一层的k缓存
        past_lengths = first_layer_k.size(2)  # 获取历史长度维度
        num_kv_heads = first_layer_k.size(1)  # 获取键值头数 [关键修改]

        # 创建扩展的注意力掩码 [batch_size, 1, seq_length, total_length]
        total_length = past_lengths + seq_length
        extended_attention_mask = torch.zeros(
            batch_size, 1, seq_length, total_length,
            dtype=torch.bool, device=input_ids.device
        )

        # 填充掩码：
        # - 历史部分：全部可见（1表示不掩盖）
        # - 当前部分：因果可见（左下三角为1）

        # 历史部分（从位置0到past_lengths）全部可见
        extended_attention_mask[:, :, :, :past_lengths] = 1

        # 当前部分：因果掩码（只允许每个token关注自身及之前的token）
        # 创建一个左下三角矩阵（包括对角线）作为当前部分的掩码
        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

        # 将因果掩码放置在从past_lengths开始的列位置
        extended_attention_mask[:, :, :, past_lengths:past_lengths + seq_length] = causal_mask

        # 扩展掩码到键值头数 [关键修改]
        extended_attention_mask = extended_attention_mask.repeat(1, num_kv_heads, 1, 1)

        # 在返回前添加连续化操作
        extended_attention_mask = extended_attention_mask.contiguous()
        return extended_attention_mask