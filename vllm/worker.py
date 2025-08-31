import torch
import time
import traceback
from typing import List, Dict, Any
from transformers import AutoTokenizer
from .cache import PagedKVCache
from .sampling.sampler import Sampler
from .models.model_registry import get_model
from .utils.tensor_parallel import TensorParallelManager
from .schema import Response


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

        # 设备设置
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = torch.device(device)

        if "cuda" in str(self.device) and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")

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
        self.model.model.to(self.device)

        num_key_value_heads = getattr(self.model.config, "num_key_value_heads",
                                      self.model.config.num_attention_heads)

        # 初始化KV缓存
        self.kv_cache = PagedKVCache(
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_size=self.model.config.hidden_size // self.model.config.num_attention_heads,
            page_size=256,
            max_num_seqs=max_num_seqs,
            memory_manager=memory_manager,
            device=self.device,
            max_seq_length=max_seq_length,
        )

        print(f"Model config - num_attention_heads: {self.model.config.num_attention_heads}")
        print(f"Model config - num_key_value_heads: {num_key_value_heads}")

        # 初始化采样器
        self.sampler = Sampler()
        self.is_running = True
        print(f"ModelWorker initialized with model: {model_name}")

    def process_batch(self, requests: List[Any]) -> List[Any]:
        print(f"Processing batch of {len(requests)} requests")
        if not self.is_running or not requests:
            return []

        try:
            # 准备输入数据
            input_data = self._prepare_inputs(requests)
            print(f"Input IDs shape: {input_data['input_ids'].shape}")

            # 执行模型前向传播
            output_data = self._forward_pass(input_data)

            # 获取最后一个token的logits
            last_logits = output_data["logits"][:, -1, :]
            print(f"Last token logits shape: {last_logits.shape}")

            # 逐个样本采样
            next_tokens_list = []
            for i in range(last_logits.size(0)):
                logits_i = last_logits[i]
                sampling_param_i = input_data["sampling_params"][i]
                next_token_i = self.sampler.sample(logits_i, sampling_param_i)
                next_tokens_list.append(next_token_i)

            # 合并结果
            next_tokens = torch.cat(next_tokens_list, dim=0)

            # 更新KV缓存
            print(f"Updating cache for {len(input_data['sequence_ids'])} sequences")
            self.kv_cache.update_cache(
                output_data["hidden_states"],
                input_data["sequence_ids"],
                input_data["positions"]
            )

            # 准备响应
            responses = self._prepare_responses(requests, next_tokens)
            print(f"Batch processed successfully")
            return responses

        except Exception as e:
            print(f"Error processing batch: {e}\n{traceback.format_exc()}")
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

        # 位置信息
        positions = torch.arange(
            input_ids.size(1),
            device=self.device
        ).unsqueeze(0).expand(input_ids.size(0), -1)
        positions = positions.to(self.device).contiguous()

        # 历史长度
        past_seq_lengths = [
            self.kv_cache.get_sequence_length(seq_id) for seq_id in sequence_ids
        ]

        return {
            "input_ids": input_ids,
            "positions": positions,
            "sequence_ids": sequence_ids,
            "sampling_params": sampling_params,
            "past_seq_lengths": past_seq_lengths
        }

    def _forward_pass(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Model device: {next(self.model.model.parameters()).device}")
        print(f"Input IDs device: {input_data['input_ids'].device}")

        # 获取历史KV缓存
        past_key_values = None
        if any(length > 0 for length in input_data["past_seq_lengths"]):
            past_key_values = self.kv_cache.get_cache(
                input_data["sequence_ids"],
                input_data["past_seq_lengths"]
            )

        # 确保past_key_values连续
        if past_key_values:
            past_key_values = [
                (k.contiguous(), v.contiguous()) for k, v in past_key_values
            ]

        # 执行模型前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_data["input_ids"].contiguous(),
                positions=input_data["positions"].contiguous(),
                past_key_values=past_key_values,
                use_cache=True
            )

        # 统一输出格式
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values", None)
            hidden_states = outputs.get("hidden_states", logits)
        else:
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                raise ValueError("Cannot get logits from model outputs")

            past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None

            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = logits

        # 确保hidden_states是张量
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = logits

        print(f"Logits shape: {logits.shape}")
        print(f"Hidden states shape: {hidden_states.shape}")

        return {
            "logits": logits,
            "past_key_values": past_key_values,
            "hidden_states": hidden_states
        }

    def _prepare_responses(self, requests: List[Any], next_tokens: torch.Tensor) -> List[Any]:
        responses = []

        for i, req in enumerate(requests):
            token_tensor = next_tokens[i].unsqueeze(0)

            try:
                generated_text = self.tokenizer.decode(
                    token_tensor,
                    skip_special_tokens=True
                )
                error_msg = None
            except Exception as e:
                generated_text = ""
                error_msg = f"Decoding error: {str(e)}"

            responses.append(Response(
                request_id=req.request_id,
                generated_text=generated_text,
                success=bool(generated_text),
                error_message=error_msg
            ))

        return responses

    def stop(self):
        self.is_running = False