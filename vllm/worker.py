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
            # 初始化响应列表
            responses = [None] * len(requests)

            # 创建序列状态跟踪
            active_sequences = {}
            for i, req in enumerate(requests):
                # 初始化序列状态
                seq_id = req.request_id
                active_sequences[seq_id] = {
                    'prompt': req.prompt,
                    'tokens': [],
                    'max_tokens': req.sampling_params.max_tokens,
                    'sampling_params': req.sampling_params,
                    'finished': False,
                    'position': 0
                }

            # 持续生成直到所有序列完成
            while any(not seq['finished'] for seq in active_sequences.values()):
                # 准备当前批次的输入
                batch_input = self._prepare_active_batch(active_sequences)
                if not batch_input:
                    break

                # 执行模型前向传播
                output_data = self._forward_pass(batch_input)

                # 处理输出并采样
                last_logits = output_data["logits"][:, -1, :]
                next_tokens = self._sample_next_tokens(
                    last_logits,
                    batch_input["sampling_params"]
                )

                # 更新序列状态
                self._update_sequences(
                    active_sequences,
                    batch_input["sequence_ids"],
                    next_tokens
                )

                # 更新KV缓存
                self.kv_cache.update_cache(
                    output_data["hidden_states"],
                    batch_input["sequence_ids"],
                    batch_input["positions"]
                )

            # 准备最终响应
            responses = self._prepare_final_responses(active_sequences, requests)
            print(
                f"Batch processed successfully. Generated {sum(len(seq['tokens']) for seq in active_sequences.values())} tokens")
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

    def _prepare_active_batch(self, active_sequences: dict) -> dict:
        """为活跃序列准备输入"""
        sequence_ids = []
        input_ids_list = []
        sampling_params = []
        positions_list = []

        for seq_id, seq_state in active_sequences.items():
            if seq_state['finished']:
                continue

            # 获取当前token位置
            current_token = seq_state['tokens'][-1] if seq_state['tokens'] else self.tokenizer.bos_token_id
            input_ids = torch.tensor([[current_token]], device=self.device)

            sequence_ids.append(seq_id)
            input_ids_list.append(input_ids)
            sampling_params.append(seq_state['sampling_params'])

            # 位置 = 当前序列长度
            position = torch.tensor([[len(seq_state['tokens'])]], device=self.device)
            positions_list.append(position)

        if not sequence_ids:
            return {}

        # 批处理输入
        input_ids = torch.cat(input_ids_list, dim=0)
        positions = torch.cat(positions_list, dim=0)

        return {
            "input_ids": input_ids,
            "positions": positions,
            "sequence_ids": sequence_ids,
            "sampling_params": sampling_params,
            "past_seq_lengths": [self.kv_cache.get_sequence_length(seq_id) for seq_id in sequence_ids]
        }

    def _sample_next_tokens(self, logits: torch.Tensor, sampling_params_list: list) -> torch.Tensor:
        """批量采样下一个token"""
        next_tokens = []
        for i in range(logits.size(0)):
            sampled_token = self.sampler.sample(logits[i], sampling_params_list[i])
            next_tokens.append(sampled_token)
        return torch.stack(next_tokens)

    def _update_sequences(self, active_sequences: dict, sequence_ids: list, next_tokens: torch.Tensor):
        """更新序列状态"""
        for i, seq_id in enumerate(sequence_ids):
            token = next_tokens[i].item()
            seq_state = active_sequences[seq_id]

            # 添加到生成的token
            seq_state['tokens'].append(token)

            # 检查是否完成
            seq_state['finished'] = (
                    len(seq_state['tokens']) >= seq_state['max_tokens'] or
                    token == self.tokenizer.eos_token_id
            )

    def _prepare_final_responses(self, active_sequences: dict, requests: list) -> list:
        """准备最终响应"""
        responses = []
        for req in requests:
            seq_state = active_sequences[req.request_id]
            try:
                generated_text = self.tokenizer.decode(
                    seq_state['tokens'],
                    skip_special_tokens=True
                )
                responses.append(Response(
                    request_id=req.request_id,
                    generated_text=generated_text,
                    success=True
                ))
            except Exception as e:
                responses.append(Response(
                    request_id=req.request_id,
                    generated_text="",
                    success=False,
                    error_message=f"Decoding error: {str(e)}"
                ))

            # 清理序列缓存
            self.kv_cache.remove_sequence(req.request_id)

        return responses

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
        # 设备检查 ↓↓↓
        print(f"Model device: {next(self.model.model.parameters()).device}")
        print(f"Input IDs device: {input_data['input_ids'].device}")
        print(f"Positions device: {input_data['positions'].device}")
        # 打印调试信息
        print(f"Input IDs shape: {input_data['input_ids'].shape}")
        print(f"Positions shape: {input_data['positions'].shape}")

        # 添加连续性检查
        print(f"Input IDs contiguous: {input_data['input_ids'].is_contiguous()}")
        print(f"Positions contiguous: {input_data['positions'].is_contiguous()}")

        """执行模型前向传播"""
        # 获取历史KV缓存
        # 获取历史KV缓存（可能为None）
        past_key_values = None
        if input_data["past_seq_lengths"].sum().item() > 0:  # 只有存在历史缓存时才获取
            past_key_values = self.kv_cache.get_cache(
                input_data["sequence_ids"],
                input_data["past_seq_lengths"]
            )

        # 检查past_key_values的连续性
        if past_key_values:
            for i, (k, v) in enumerate(past_key_values):
                print(f"Layer {i} k contiguous: {k.is_contiguous()}")
                print(f"Layer {i} v contiguous: {v.is_contiguous()}")
        if past_key_values is not None:
            print(f"Past KV length: {len(past_key_values)}")
            print(f"First layer k shape: {past_key_values[0][0].shape}")
            print(f"First layer v shape: {past_key_values[0][1].shape}")
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

        # 确保past_key_values连续
        if past_key_values is not None:
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

        # ======== 修复点：使用字典键访问而不是属性访问 ========
        if isinstance(outputs, dict):
            # 使用字典键访问
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values", None)

            # 获取隐藏状态
            hidden_states = outputs.get("hidden_states", logits)

            # 如果hidden_states是元组（包含所有层），则取最后一层
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[-1]
        else:
            # 处理其他输出类型（如元组）
            print(f"Unexpected output type: {type(outputs)}")
            # 尝试获取logits和past_key_values
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # 如果连logits属性都没有，则尝试第一个元素
                if isinstance(outputs, tuple) and len(outputs) > 0:
                    logits = outputs[0]
                else:
                    raise ValueError("Cannot get logits from model outputs")

            past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None

            # 尝试获取隐藏状态
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                print("Warning: Using logits as fallback for hidden_states")
                hidden_states = logits

            # 如果hidden_states是元组，取最后一层
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[-1]

        # 确保hidden_states是张量（重要！）
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = logits  # 回退使用logits
        # 打印确认形状
        print(f"Logits shape: {logits.shape}")
        print(f"Hidden states shape: {hidden_states.shape}")

        # 确保返回三个关键值
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