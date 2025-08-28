import torch
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer
from .cache import PagedKVCache
from .sampling.sampler import Sampler
from .models.model_registry import get_model
from .utils.tensor_parallel import TensorParallelManager
from .schema import Response  # 从schema导入

class ModelWorker:
    """模型工作器，负责实际的前向传播和生成"""

    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 tensor_parallel_size: int,
                 max_num_seqs: int,
                 max_seq_length: int,
                 device: str,
                 memory_manager):
        self.model_name = model_name
        self.device = device
        self.max_num_seqs = max_num_seqs
        self.max_seq_length = max_seq_length
        self.memory_manager = memory_manager

        # 修复设备设置 (添加以下代码)
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"

        self.device = torch.device(device)

        # 替换原有的设置代码
        if "cuda" in str(self.device):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available")
            print( torch.cuda.is_available())
            torch.cuda.set_device(self.device)



        # 强制设置PyTorch默认设备
        torch.cuda.set_device(self.device)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化张量并行管理器
        self.tensor_parallel_manager = TensorParallelManager(tensor_parallel_size)

        # 加载模型
        self.model = get_model(model_name).initialize(
            model_name=model_name,
            tensor_parallel_manager=self.tensor_parallel_manager,
            memory_manager=memory_manager
        )
        # +++ 添加模型迁移到GPU +++
        self.model.model.to(self.device)  # 确保模型迁移到指定设备
        num_key_value_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        # 初始化KV缓存
        self.kv_cache = PagedKVCache(
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,  # 关键添加
            head_size=self.model.config.hidden_size // self.model.config.num_attention_heads,
            page_size=256,  # 每页256个token
            max_num_seqs=max_num_seqs,
            memory_manager=memory_manager,
            device = self.device,  # 确保传递当前设备
            max_seq_length=max_seq_length,  # 传入最大序列长度

        )

        # 初始化采样器
        self.sampler = Sampler()

        # 运行状态
        self.is_running = True

        print(f"ModelWorker initialized with model: {model_name}")

    def process_batch(self, requests: List[Any]) -> List[Any]:

        print(f"Processing batch of {len(requests)} requests")
        """处理一批请求"""
        if not self.is_running or not requests:
            return []

        try:
            # 准备输入数据
            input_data = self._prepare_inputs(requests)

            # 执行模型前向传播
            output_data = self._forward_pass(input_data)

            # 采样生成下一个token
            next_tokens = self.sampler.sample(output_data["logits"], input_data["sampling_params"])

            # 更新KV缓存
            self.kv_cache.update_cache(
                output_data["hidden_states"],
                input_data["sequence_ids"],
                input_data["positions"]
            )

            # 准备响应
            responses = self._prepare_responses(requests, next_tokens)
            print(f"Batch processed successfully. Response type: {type(responses[0])}")
            return responses

        except Exception as e:
            print(f"Error processing batch: {e}")
            # 返回Response对象列表
            return [
                Response(
                    request_id=req.request_id,
                    generated_text="",
                    success=False,
                    error_message=str(e)
                )
                for req in requests
            ]

    def _prepare_inputs(self, requests: List[Any]) -> Dict[str, Any]:
        """准备输入数据"""
        prompts = [req.prompt for req in requests]
        sampling_params = [req.sampling_params for req in requests]
        sequence_ids = [req.request_id for req in requests]

        # 分词
        input_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length
        ).input_ids.to(self.device)

        # 获取位置信息
        positions = torch.arange(
            input_ids.size(1),
            device=self.device
        ).unsqueeze(0).expand(input_ids.size(0), -1)
        positions = positions.to(self.device)

        # 获取历史长度（对于连续生成）


        past_seq_lengths = torch.tensor([
            self.kv_cache.get_sequence_length(seq_id) for seq_id in sequence_ids
        ], device=self.device)  # 确保在GPU上
        return {
            "input_ids": input_ids,
            "positions": positions,
            "sequence_ids": sequence_ids,
            "sampling_params": sampling_params,
            "past_seq_lengths": past_seq_lengths
        }

    def _forward_pass(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 设备检查 ↓↓↓
        print(f"Model device: {next(self.model.model.parameters()).device}")
        print(f"Input IDs device: {input_data['input_ids'].device}")
        print(f"Positions device: {input_data['positions'].device}")


        """执行模型前向传播"""
        # 获取历史KV缓存
        # 获取历史KV缓存（可能为None）
        past_key_values = None
        if input_data["past_seq_lengths"].sum().item() > 0:  # 只有存在历史缓存时才获取
            past_key_values = self.kv_cache.get_cache(
                input_data["sequence_ids"],
                input_data["past_seq_lengths"]
            )

        # 打印调试信息
        print(f"Past KV is None: {past_key_values is None}")
        print(f"Past KV type: {type(past_key_values)}")
        if past_key_values and len(past_key_values) > 0:
            first_tensor = past_key_values[0][0]
            print(f"First KV tensor device: {first_tensor.device}")
            print(f"First layer type: {type(past_key_values[0])}")
            if len(past_key_values[0]) > 0:
                print(f"First layer k type: {type(past_key_values[0][0])}")
                if len(past_key_values[0][0]) > 0:
                    print(f"First sequence k type: {type(past_key_values[0][0][0])}")
                    print(
                        f"First sequence k shape: {past_key_values[0][0][0].shape if hasattr(past_key_values[0][0][0], 'shape') else 'No shape'}")

        # 执行模型前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_data["input_ids"],
                positions=input_data["positions"],
                past_key_values=past_key_values,
                use_cache=True
            )

        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values
        }

    def _prepare_responses(self, requests: List[Any], next_tokens: torch.Tensor) -> List[Any]:
        """准备响应数据"""
        responses = []
        # 确保requests和next_tokens长度匹配
        # 检查张量维度是否匹配
        if next_tokens.size(0) != len(requests):
            print(f"Warning: Token count {next_tokens.size(0)} != request count {len(requests)}")
            # 填充缺失的token
            next_tokens = torch.cat([
                next_tokens,
                torch.full((len(requests) - next_tokens.size(0),), -1, device=self.device)
            ])

        for i, req in enumerate(requests):
            token_tensor = next_tokens[i].unsqueeze(0)

            # 检查token有效性
            if token_tensor.item() == -1:
                responses.append(Response(
                    request_id=req.request_id,
                    generated_text="",
                    success=False,
                    error_message="Token generation failed"
                ))
                continue

            try:
                # 安全解码
                generated_text = self.tokenizer.decode(
                    token_tensor,
                    skip_special_tokens=True
                )
            except Exception as e:
                generated_text = ""
                error_msg = f"Decoding error: {str(e)}"
            else:
                error_msg = None

            responses.append(Response(
                request_id=req.request_id,
                generated_text=generated_text,
                success=(generated_text != ""),
                error_message=error_msg
            ))
        return responses

    def stop(self):
        """停止工作器"""
        self.is_running = False