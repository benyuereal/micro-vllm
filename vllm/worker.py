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

        # 修复设备设置
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
            device=self.device,  # 确保传递当前设备
            max_seq_length=max_seq_length,  # 传入最大序列长度
        )

        print(f"Model config - num_attention_heads: {self.model.config.num_attention_heads}")
        print(f"Model config - num_key_value_heads: {num_key_value_heads}")

        # 初始化采样器
        self.sampler = Sampler()

        # 运行状态
        self.is_running = True

        print(f"ModelWorker initialized with model: {model_name}")

    def process_batch(self, requests: List[Any]) -> List[Any]:
        print(f"Processing batch of {len(requests)} requests")
        if not self.is_running or not requests:
            return []

        try:
            # 调试信息：显示当前序列状态
            print("Current batch sequence states:")
            for req in requests:
                seq_length = self.kv_cache.get_sequence_length(req.request_id)
                print(f"Request {req.request_id}: "
                      f"Prompt len={len(req.prompt_ids)}, "
                      f"Generated tokens={len(getattr(req, 'generated_token_ids', []))}, "
                      f"Cache length={seq_length}")

            # 维护每个请求的生成状态
            for req in requests:
                if not hasattr(req, 'generated_token_ids'):
                    req.generated_token_ids = []  # 存储已生成的token IDs
                    req.prompt_ids = self.tokenizer.encode(
                        req.prompt,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=self.max_seq_length
                    ).to(self.device).squeeze(0)
                    req.remaining_tokens = req.sampling_params.max_tokens
                    req.is_completed = False
                    req.start_time = time.time()
                    req.last_decoded_index = 0  # 追踪最后解码位置
                    # 初始化KV缓存
                    self.kv_cache.add_sequence(req.request_id)

            # 准备当前步的输入
            input_ids_batch = []
            positions_batch = []
            sequence_ids = []
            past_seq_lengths = []
            sampling_params_list = []

            for req in requests:
                if req.is_completed:
                    continue

                if len(req.generated_token_ids) == 0:
                    # 第一次生成：使用完整prompt
                    input_ids = req.prompt_ids
                    positions = torch.arange(len(input_ids), device=self.device)
                else:
                    # 后续生成：只使用最新生成的token
                    input_ids = torch.tensor([req.generated_token_ids[-1]], device=self.device)
                    positions = torch.tensor([len(req.prompt_ids) + len(req.generated_token_ids) - 1],
                                             device=self.device)

                input_ids_batch.append(input_ids)
                positions_batch.append(positions)
                sequence_ids.append(req.request_id)
                past_seq_lengths.append(self.kv_cache.get_sequence_length(req.request_id))
                sampling_params_list.append(req.sampling_params)

            # 填充batch
            max_len = max(len(ids) for ids in input_ids_batch)
            padded_input_ids = torch.full((len(input_ids_batch), max_len), self.tokenizer.pad_token_id,
                                          device=self.device)
            padded_positions = torch.full((len(positions_batch), max_len), -1, device=self.device)

            for i, (ids, pos) in enumerate(zip(input_ids_batch, positions_batch)):
                padded_input_ids[i, :len(ids)] = ids
                if len(pos) > 0:  # 确保位置张量非空
                    padded_positions[i, :len(pos)] = pos

            # 准备输入数据
            input_data = {
                "input_ids": padded_input_ids,
                "positions": padded_positions,
                "sequence_ids": sequence_ids,
                "sampling_params": sampling_params_list,
                "past_seq_lengths": torch.tensor(past_seq_lengths, device=self.device)
            }

            # 执行前向传播
            output_data = self._forward_pass(input_data)

            # 获取最后一个token的logits
            last_logits = output_data["logits"][:, -1, :]
            print(f"Last token logits shape: {last_logits.shape}")

            # 逐个样本采样
            next_tokens_list = []
            for i, req in enumerate(requests):
                if req.is_completed:
                    next_tokens_list.append(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
                    continue

                logits_i = last_logits[i]
                next_token = self.sampler.sample(logits_i.unsqueeze(0), req.sampling_params)
                next_tokens_list.append(next_token)

                # 更新请求状态
                req.generated_token_ids.append(next_token.item())
                req.remaining_tokens -= 1

                # 检查完成条件
                if (req.remaining_tokens <= 0 or
                        next_token.item() == self.tokenizer.eos_token_id or
                        time.time() - req.start_time > 30):  # 超时保护
                    req.is_completed = True

            # 更新KV缓存
            if output_data["hidden_states"] is not None:
                # 确保形状兼容
                hidden_states = output_data["hidden_states"]
                if hidden_states.dim() == 3:  # [batch, seq, dim]
                    self.kv_cache.update_cache(
                        hidden_states,  # 使用全部隐藏状态
                        sequence_ids,
                        padded_positions
                    )
                else:
                    print(f"Warning: Unexpected hidden states shape {hidden_states.shape}")

            # 准备响应
            responses = []
            for req in requests:
                if req.is_completed:
                    # 完整解码生成的文本
                    full_ids = torch.cat([
                        req.prompt_ids,
                        torch.tensor(req.generated_token_ids, device=self.device)
                    ])
                    generated_text = self.tokenizer.decode(
                        full_ids[len(req.prompt_ids):],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    responses.append(Response(
                        request_id=req.request_id,
                        generated_text=generated_text,
                        success=True,
                        error_message=None
                    ))
                    # 清理缓存
                    self.kv_cache.remove_sequence(req.request_id)
                else:
                    # 获取自上次响应后新生成的token
                    new_tokens = req.generated_token_ids[req.last_decoded_index:]
                    # 更新最后解码位置
                    req.last_decoded_index = len(req.generated_token_ids)

                    # 累积解码新token
                    new_text = self.tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )

                    responses.append(Response(
                        request_id=req.request_id,
                        generated_text=new_text,
                        success=False,
                        error_message=None
                    ))

            print(
                f"Batch processed successfully. Completed requests: {sum(1 for r in requests if r.is_completed)}/{len(requests)}")
            return responses

        except Exception as e:
            import traceback
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
        positions = positions.to(self.device).contiguous()

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
        """执行模型前向传播（完整修复版）"""
        # 打印调试信息
        print(f"Input IDs shape: {input_data['input_ids'].shape}")
        print(f"Positions shape: {input_data['positions'].shape}")

        # 确保张量连续
        input_ids = input_data["input_ids"].contiguous()
        positions = input_data["positions"].contiguous()

        # 获取历史KV缓存
        past_key_values = None
        if input_data["past_seq_lengths"].sum().item() > 0:
            try:
                # 转换past_seq_lengths为列表
                past_lengths_list = input_data["past_seq_lengths"].cpu().tolist()
                past_key_values = self.kv_cache.get_cache(
                    input_data["sequence_ids"],
                    past_lengths_list
                )

                # 确保缓存张量连续
                if past_key_values:
                    past_key_values = [
                        (k.contiguous(), v.contiguous())
                        for k, v in past_key_values
                    ]
            except Exception as e:
                print(f"Error getting KV cache: {e}")
                past_key_values = None

        # 打印缓存信息
        print(f"Past KV is None: {past_key_values is None}")
        if past_key_values:
            print(f"Past KV length: {len(past_key_values)}")
            if len(past_key_values) > 0:
                first_layer = past_key_values[0]
                if first_layer and len(first_layer) > 0:
                    k, v = first_layer
                    print(f"First layer k shape: {k.shape}")
                    print(f"First layer v shape: {v.shape}")

        # 执行模型前向传播
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            except Exception as e:
                print(f"Model forward error: {e}")
                # 尝试不使用缓存再次运行
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    past_key_values=None,
                    use_cache=False
                )
                past_key_values = None

        # 统一处理输出格式
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values", None)
            hidden_states = outputs.get("hidden_states", logits)
        else:
            # 处理其他输出类型（如元组）
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    raise ValueError("Cannot get logits from model outputs")

            past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else logits

        # 确保hidden_states是张量
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = logits

        # 打印输出形状
        print(f"Logits shape: {logits.shape}")
        print(f"Hidden states shape: {hidden_states.shape}")

        return {
            "logits": logits,
            "past_key_values": past_key_values,
            "hidden_states": hidden_states
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