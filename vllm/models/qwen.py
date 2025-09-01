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

        # 修改输出处理逻辑
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True
        )

        # 统一输出格式
        if not isinstance(outputs, dict):
            # 处理元组输出
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, past_key_values = outputs
                hidden_states = logits  # 默认使用logits
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")
        else:
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # 优先获取last_hidden_state（最终隐藏层）
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            # 次选hidden_states（需要取最后一层）
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]  # 取最后一层
            else:
                hidden_states = logits
                print("Warning: Using logits as fallback for hidden_states")

        # +++ 关键修复：返回字典必须包含 hidden_states +++
        # 确保字典包含 hidden_states
        print("hidden_states:", hidden_states.shape)
        return {
            "logits": logits,
            "past_key_values": past_key_values,
            "hidden_states": hidden_states
        }

    # qwen.py 中的 _prepare_attention_mask 方法
    def _prepare_attention_mask(self,
                                input_ids: torch.Tensor,
                                past_key_values: Optional[Tuple] = None) -> torch.Tensor:
        """准备符合Qwen要求的注意力掩码"""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # 没有历史缓存时创建简单掩码
        if past_key_values is None:
            return torch.ones(batch_size, seq_length, dtype=torch.long, device=device)

        # 获取历史长度
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer_k = past_key_values[0][0]
            if first_layer_k is not None:
                # 张量形状: [batch, num_heads, seq_len, head_dim]
                past_length = first_layer_k.size(2)

        # 创建扩展掩码 [batch_size, total_length]
        total_length = past_length + seq_length
        attention_mask = torch.ones(batch_size, total_length, dtype=torch.long, device=device)

        # 创建因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones(total_length, total_length, device=device)).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1) & causal_mask

        # 压缩维度: [batch_size, 1, total_length, total_length] -> [batch_size, total_length]
        attention_mask = attention_mask[:, 0, 0]  # 取第一个token的掩码作为代表

        return attention_mask.contiguous()