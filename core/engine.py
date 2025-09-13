import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator, Optional
from queue import Queue
import time
import math

from models.qwen_adapter import QwenModelAdapter
from .memory_manager import MemoryPool
from .scheduler import Scheduler
from .cache_manager import KVCache
from .sequence import Sequence
from .model_loader import load_model
from .paged_attention import PagedAttention


class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()

        # 获取模型配置
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_size = self.model.config.hidden_size // num_heads
        print("load model config:", self.model.config)
        # 检查是否有num_key_value_heads属性，如果没有则使用num_heads
        if hasattr(self.model.config, 'num_key_value_heads'):
            num_key_value_heads = self.model.config.num_key_value_heads
        else:
            num_key_value_heads = num_heads

        # 自动检测设备并优化配置
        if torch.backends.mps.is_available():
            device = 'mps'
            block_size = 16
            max_blocks = 512
            dtype = torch.float16
            # self.model = self.model.to(torch.float32)
        elif torch.cuda.is_available():
            device = 'cuda'
            block_size = 64
            max_blocks = 640  # 40GB / (64 * 128 * 32 * 2 * 2) ≈ 640
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            device = 'cpu'
            block_size = 16
            max_blocks = 128
            dtype = torch.float32

        # 初始化内存池和KV缓存
        model_dtype = next(self.model.parameters()).dtype
        self.memory_pool = MemoryPool(
            block_size=block_size,
            max_blocks=max_blocks,
            num_layers=num_layers,
            num_heads=num_key_value_heads,
            head_size=head_size,
            dtype=dtype
        )

        self.cache = KVCache(self.memory_pool)
        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.adapter = QwenModelAdapter()

        # 初始化PageAttention模块
        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=num_key_value_heads,
            device=device
        )

        self.device = device
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stream_callbacks = {}
        self.max_position = 4096  # 最大位置嵌入

    def add_request(self, prompt: str, max_tokens: int = 128,
                    temperature: float = 0.7, top_p: float = 0.9, priority: int = 0) -> int:
        """添加生成请求，返回序列ID"""
        seq_id = hash(prompt + str(time.time())) % (2 ** 32)
        seq = Sequence(seq_id, prompt, self.tokenizer, max_tokens)
        seq.temperature = temperature
        seq.top_p = top_p
        seq.priority = priority
        self.scheduler.add_request(seq)
        return seq_id

    def register_stream_callback(self, seq_id: int, callback):
        """注册流式回调函数"""
        self.stream_callbacks[seq_id] = callback

    def unregister_stream_callback(self, seq_id: int):
        """取消注册流式回调函数"""
        self.stream_callbacks.pop(seq_id, None)

    def _invoke_stream_callback(self, seq_id: int, token: int, text: str):
        """调用流式回调函数"""
        if callback := self.stream_callbacks.get(seq_id):
            try:
                callback(token, text)
            except Exception as e:
                print(f"Error in stream callback for seq {seq_id}: {e}")

    @torch.no_grad()
    def step(self) -> bool:
        """执行一个推理step，返回是否有处理任何请求"""
        batch, batch_type = self.scheduler.get_next_batch()
        if not batch:
            return False

        seq_ids = [seq.seq_id for seq in batch]
        context_lens = [seq.current_position - 1 for seq in batch]

        if batch_type == "prefill":
            self._process_prefill_batch(batch, seq_ids, context_lens)
        elif batch_type == "decode":
            self._process_decode_batch(batch, seq_ids, context_lens)

        return True

    def _process_prefill_batch(self, batch: List[Sequence], seq_ids: List[int], context_lens: List[int]):
        """处理预填充批次"""
        input_ids_list = [seq.get_next_input_ids() for seq in batch]
        input_tensor = self._pad_batch(input_ids_list, self.tokenizer.pad_token_id).to(self.device)

        # Prefill阶段使用标准模型实现
        outputs = self.model(input_ids=input_tensor, use_cache=True)
        logits, past_key_values = self.adapter.process_outputs(outputs, input_tensor.size(1))

        # 分配KV缓存
        for i, seq in enumerate(batch):
            # 提取序列的token和位置
            token_positions = []
            for pos, token_id in enumerate(seq.get_next_input_ids()):
                token_positions.append((token_id, pos))

            # 提取该序列的KV数据
            num_layers = self.memory_pool.num_layers
            num_heads = self.memory_pool.num_heads
            head_size = self.memory_pool.head_size
            seq_length = len(token_positions)

            # 初始化KV缓存张量
            k_cache = torch.zeros(
                (num_layers, num_heads, seq_length, head_size),
                dtype=self.memory_pool.dtype, device=self.device
            )
            v_cache = torch.zeros_like(k_cache)

            # 填充KV缓存 past_key_values.layers= 24 x [batch_size,num_heads, seq_length,head_size]
            for layer_idx in range(num_layers):
                layer_k = past_key_values[layer_idx][0][i:i + 1]  # [batch=1, num_heads, seq_len, head_size]
                layer_v = past_key_values[layer_idx][1][i:i + 1]

                k_cache[layer_idx] = layer_k.squeeze(0).permute(1, 0, 2)
                v_cache[layer_idx] = layer_v.squeeze(0).permute(1, 0, 2)

            # 为序列分配缓存 [num_layers,num_heads, seq_length,head_size]
            self.cache.allocate(seq.seq_id, token_positions, k_cache, v_cache)

            # 采样下一个token
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            self._update_sequence(seq, next_token)

    def _process_decode_batch(self, batch: List[Sequence], seq_ids: List[int], context_lens: List[int]):
        """处理解码批次"""
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)

        token_positions = []
        for seq in batch:
            tokens = self.cache.get_sequence_tokens(seq.seq_id)
            positions = [min(pos, self.max_position - 1) for _, pos in tokens]
            token_positions.append(positions)

        # 修改这里：直接使用 self.model.transformer.wte 而不是 self.model.model.embed_tokens
        hidden_states = self.model.transformer.wte(input_ids)
        batch_size = len(input_ids)
        # 初始化列表以存储每层的Key和Value
        new_ks = []  # 每个元素形状: [batch_size, kv_num_heads, head_size]
        new_vs = []  # 每个元素形状: [batch_size, kv_num_heads, head_size]

        # 通过模型层
        for layer_idx, layer in enumerate(self.model.transformer.h):  # 注意这里也改为 transformer.h
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)  # Qwen 使用 ln_1 而不是 input_layernorm

            # 计算Query、Key、Value
            query = layer.attn.c_attn(hidden_states)
            # Qwen 的 c_attn 是 q,k,v 的合并投影，需要拆分
            query, key, value = query.split(self.model.config.hidden_size, dim=2)

            # 获取头数和头尺寸
            head_size = self.paged_attention.head_size
            num_heads = self.paged_attention.num_heads
            kv_num_heads = self.paged_attention.kv_num_heads

            # 重塑Query、Key、Value
            query = query.view(batch_size, num_heads, head_size)
            key = key.view(batch_size, kv_num_heads, head_size)
            value = value.view(batch_size, kv_num_heads, head_size)

            # 保存Key和Value用于后续缓存
            new_ks.append(key)
            new_vs.append(value)

            # 获取当前token的位置（每个序列的当前位置）
            current_positions = [seq.current_position - 1 for seq in batch]

            # 调用PagedAttention
            attn_output = self.paged_attention.forward(
                query=query,
                cache_manager=self.cache,
                seq_ids=seq_ids,
                context_lens=context_lens,
                token_positions=token_positions,
                layer_idx=layer_idx,
                current_k=key,
                current_v=value,
                current_positions=current_positions
            )

            attn_output = attn_output.reshape(attn_output.size(0), -1)
            attn_output = layer.attn.c_proj(attn_output)  # Qwen 使用 c_proj 而不是 o_proj
            attn_output = attn_output.unsqueeze(1)
            hidden_states = residual + attn_output

            # 前馈网络
            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)  # Qwen 使用 ln_2
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.model.transformer.ln_f(hidden_states)  # Qwen 使用 ln_f
        logits = self.model.lm_head(hidden_states).float()

        # 采样下一个token
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            next_tokens.append(next_token)

        # 存储当前输入token的KV到缓存中
        for i, seq in enumerate(batch):
            input_token_id = input_ids[i].item()
            input_token_position = seq.current_position - 1

            num_layers = self.memory_pool.num_layers
            num_heads = self.memory_pool.num_heads
            head_size = self.memory_pool.head_size

            k_cache = torch.zeros((num_layers, num_heads, 1, head_size), dtype=self.memory_pool.dtype,
                                  device=self.device)
            v_cache = torch.zeros_like(k_cache)

            for layer_idx in range(num_layers):
                k = new_ks[layer_idx][i]
                v = new_vs[layer_idx][i]
                k_cache[layer_idx] = k.view(1, num_heads, 1, head_size)
                v_cache[layer_idx] = v.view(1, num_heads, 1, head_size)

            self.cache.allocate(
                seq.seq_id,
                [(input_token_id, input_token_position)],
                k_cache,
                v_cache
            )

            self._update_sequence(seq, next_tokens[i])


    def _update_sequence(self, seq: Sequence, next_token: int):
        """更新序列状态"""
        seq.update_state(next_token, None)

        # 流式输出回调
        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        self._invoke_stream_callback(seq.seq_id, next_token, token_text)

        if seq.is_finished():
            self.cache.delete(seq.seq_id)
            self.scheduler.mark_finished(seq)

    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
        """填充批次"""
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """采样下一个token"""
        logits = logits / temperature

        # Top-p 采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def stream_generate(self, prompt: str, max_tokens: int = 128,
                        temperature: float = 0.7, top_p: float = 0.9) -> Generator[Tuple[int, str], None, None]:
        """流式生成"""
        seq_id = self.add_request(prompt, max_tokens, temperature, top_p)
        token_queue = Queue()

        def callback(token, text):
            token_queue.put((token, text))

        self.register_stream_callback(seq_id, callback)

        try:
            generated_count = 0
            while generated_count < max_tokens:
                if self.step():
                    # 处理新生成的token
                    while not token_queue.empty():
                        token, text = token_queue.get()
                        yield token, text
                        generated_count += 1

                        if token == self.eos_token_id:
                            return

                # 检查序列状态
                if not any(seq.seq_id == seq_id for seq in self.scheduler.running_sequences):
                    break

                time.sleep(0.001)
        finally:
            self.unregister_stream_callback(seq_id)

    def generate(self, prompts: List[str], max_tokens: int = 100) -> Dict[Sequence, str]:
        """批量生成"""
        # 添加所有请求
        seq_ids = [self.add_request(prompt, max_tokens) for prompt in prompts]
        seq_map = {seq_id: prompt for seq_id, prompt in zip(seq_ids, prompts)}

        # 执行推理直到完成
        for _ in range(max_tokens):
            if not self.step() and not self.scheduler.running_sequences:
                break

        # 处理结果
        results = {}
        finished_results = self.scheduler.get_finished_results()
        for seq, output_ids in finished_results:
            try:
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                results[seq_map[seq.seq_id]] = text
            except Exception:
                results[seq_map[seq.seq_id]] = f"[Error] {output_ids}"

        # 清理未完成的序列
        for seq in self.scheduler.running_sequences:
            self.cache.delete(seq.seq_id)
        self.scheduler.running_sequences.clear()

        return results

    def cancel_request(self, seq_id: int):
        """取消请求"""
        self.scheduler.cancel_request(seq_id)
        self.cache.delete(seq_id)
        self.unregister_stream_callback(seq_id)