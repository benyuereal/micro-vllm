import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator, Optional
from queue import Queue
import time
import math

from models.qwen_adapter import QwenModelAdapter
from .layer.layer import ModelLayerAdapter
from .memory_manager import MemoryPool
from .scheduler import Scheduler
from .cache_manager import KVCacheManager, store_kvcache
from .sequence import Sequence
from .model_loader import load_model
from .paged_attention import PagedAttention
import logging


class InferenceEngine:
    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()

        # 获取模型配置
        self.model_config = self.model.config
        self.model_type = self.model_config.model_type

        # 根据模型类型设置不同的参数
        if self.model_type == "qwen":  # Qwen 7B
            num_layers = self.model_config.num_hidden_layers
            num_heads = self.model_config.num_attention_heads
            head_size = self.model_config.hidden_size // num_heads
            num_key_value_heads = getattr(self.model_config, 'num_key_value_heads', num_heads)
            self.embedding_layer = self.model.transformer.wte
            self.norm_layer = self.model.transformer.ln_f
            self.model_layers = self.model.transformer.h
        elif self.model_type == "qwen2":  # Qwen 1.5 0.5B
            num_layers = self.model_config.num_hidden_layers
            num_heads = self.model_config.num_attention_heads
            head_size = self.model_config.hidden_size // num_heads
            num_key_value_heads = self.model_config.num_key_value_heads
            self.embedding_layer = self.model.model.embed_tokens
            self.norm_layer = self.model.model.norm
            self.model_layers = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # 自动检测设备并优化配置
        if torch.backends.mps.is_available():
            device = 'mps'
            block_size = 16
            max_blocks = 512
            dtype = torch.float16
            self.model = self.model.to(torch.float16)
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

        self.block_size = block_size
        self.num_key_value_heads = num_key_value_heads
        self.head_size = head_size

        # 初始化KV缓存管理器
        self.cache_manager = KVCacheManager(
            max_blocks, block_size, num_layers, num_key_value_heads, head_size, dtype, device
        )

        # 初始化模型层适配器
        self.layer_adapter = ModelLayerAdapter(self.model_config,
                                               device=device,
                                               num_heads=num_heads,
                                               kv_num_heads= num_key_value_heads,
                                               head_size=head_size)

        # 初始化注意力模块
        self.paged_attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=num_key_value_heads,
            device=device
        )

        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.adapter = QwenModelAdapter()

        self.device = device
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stream_callbacks = {}

        # 根据模型类型设置最大位置
        if hasattr(self.model_config, 'max_position_embeddings'):
            self.max_position = self.model_config.max_position_embeddings
        else:
            self.max_position = 4096  # 默认值

        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("inference_perf.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("InferenceEngine")
        self.logger.info(f"Initializing Inference Engine for model type: {self.model_type}")

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
            self._process_prefill_batch(batch)
        elif batch_type == "decode":
            self._process_decode_batch(batch)

        return True

    @torch.no_grad()
    def _process_prefill_batch(self, batch: List[Sequence]):
        """处理预填充批次，适配不同模型架构"""
        input_ids_list = [seq.get_next_input_ids() for seq in batch]
        input_tensor = self._pad_batch(input_ids_list, self.tokenizer.pad_token_id).to(self.device)

        # 前向传播获取KV - 使用标准模型实现
        outputs = self.model(input_ids=input_tensor, use_cache=True)

        logits = outputs.logits
        past_key_values = outputs.past_key_values
        print("k shape", past_key_values[0][0].shape)
        # 存储KV到缓存
        for i, seq in enumerate(batch):
            num_tokens = len(seq.get_next_input_ids())

            # 分配缓存块并获取slot映射
            success, slot_mapping = self.cache_manager.allocate(seq.seq_id, num_tokens)
            if not success:
                continue

            # 存储每个token的KV
            token_kv = []
            for token_idx in range(num_tokens):
                # 获取当前token在所有层的KV
                layer_kv = []
                for layer_idx in range(len(past_key_values)):
                    k = past_key_values[layer_idx][0]
                    v = past_key_values[layer_idx][1]
                    if self.model_type == "qwen":
                        # Qwen 7B的KV格式
                        print(k.shape)
                        k = k.transpose(0, 1)
                        v = v.transpose(0, 1)
                        k = k[i, :, token_idx:token_idx + 1, :]
                        v = v[i, :, token_idx:token_idx + 1, :]
                        print("after", k.shape)

                    else:
                        # Qwen 1.5的KV格式
                        k = k[i, :, token_idx:token_idx + 1, :]
                        v = v[i, :, token_idx:token_idx + 1, :]
                    layer_kv.append((k, v))
                token_kv.append(layer_kv)

            # 存储到缓存
            print("layer kv shape", token_kv[0][0][0].shape)
            self._store_token_kv(seq.seq_id, token_kv, slot_mapping)

        # 采样下一个token
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            next_tokens.append(next_token)
            self._update_sequence(seq, next_token)

        return next_tokens

    def _store_token_kv(self, seq_id, token_kv, slot_mapping):
        """存储多个token的KV到缓存（按层批量存储）"""
        num_tokens = len(token_kv)
        if num_tokens == 0:
            return

        # 将slot_mapping转为tensor
        slot_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

        # 按层处理
        num_layers = len(token_kv[0])
        for layer_idx in range(num_layers):
            # 收集当前层的所有token的K和V
            k_list = []
            v_list = []
            for token_idx in range(num_tokens):
                k, v = token_kv[token_idx][layer_idx]
                # 移除seq_len维度
                k = k.squeeze(1)  # [num_kv_heads, head_dim]
                v = v.squeeze(1)  # [num_kv_heads, head_dim]
                k_list.append(k)
                v_list.append(v)

            # 堆叠成三维张量: [num_tokens, num_kv_heads, head_dim]
            k_tensor = torch.stack(k_list, dim=0)  # 形状: [num_tokens, num_kv_heads, head_dim]
            v_tensor = torch.stack(v_list, dim=0)  # 形状: [num_tokens, num_kv_heads, head_dim]

            # 获取当前层的缓存
            k_cache = self.cache_manager.get_k_cache(layer_idx)
            v_cache = self.cache_manager.get_v_cache(layer_idx)

            # 存储到缓存（现在满足三维要求）
            store_kvcache(k_tensor, v_tensor, k_cache, v_cache, slot_tensor, self.block_size, self.num_key_value_heads, self.head_size )

    @torch.no_grad()
    def _process_decode_batch(self, batch: List[Sequence]):
        """处理解码批次，适配不同模型架构"""
        self.logger.info(f"Starting decode batch with {len(batch)} sequences")
        start_time = time.perf_counter()

        # 准备输入数据
        prepare_start = time.perf_counter()
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)
        context_lens = [seq.current_position - 1 for seq in batch]
        token_positions = [[pos for pos in range(seq.current_position - 1)] for seq in batch]
        seq_ids = [seq.seq_id for seq in batch]
        prepare_time = time.perf_counter() - prepare_start
        self.logger.info(f"Data preparation time: {prepare_time * 1000:.2f}ms")

        # 嵌入层 - 使用模型特定的嵌入层
        embed_start = time.perf_counter()
        hidden_states = self.embedding_layer(input_ids)
        embed_time = time.perf_counter() - embed_start
        self.logger.info(f"Embedding time: {embed_time * 1000:.2f}ms")

        # 逐层处理
        all_layer_kvs = []
        total_layer_time = 0

        for layer_idx, layer in enumerate(self.model_layers):
            layer_start = time.perf_counter()

            # 使用模型层适配器处理不同架构的层
            hidden_states, layer_kv = self.layer_adapter.process_layer(
                layer, hidden_states, self.cache_manager, seq_ids,
                context_lens, token_positions, layer_idx,
                [seq.current_position - 1 for seq in batch]
            )

            all_layer_kvs.append(layer_kv)

            layer_time = time.perf_counter() - layer_start
            total_layer_time += layer_time
            self.logger.info(f"Layer {layer_idx} time: {layer_time * 1000:.2f}ms")

        # 最终层归一化 - 使用模型特定的归一化层
        norm_start = time.perf_counter()
        hidden_states = self.norm_layer(hidden_states)
        logits = self.model.lm_head(hidden_states).float()
        norm_time = time.perf_counter() - norm_start
        self.logger.info(f"Final norm & output time: {norm_time * 1000:.2f}ms")

        # 采样下一个token
        sample_start = time.perf_counter()
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            next_tokens.append(next_token)
        sample_time = time.perf_counter() - sample_start
        self.logger.info(f"Sampling time: {sample_time * 1000:.2f}ms")

        # KV缓存存储
        kv_store_start = time.perf_counter()
        for i, seq in enumerate(batch):
            token_idx = seq.current_position - 1
            slot = self.cache_manager.append_token(seq.seq_id, token_idx)
            if slot >= 0:
                layer_kv = []
                for layer_idx, (k, v) in enumerate(all_layer_kvs):
                    k_current = k[i:i + 1].squeeze(0).unsqueeze(1)
                    v_current = v[i:i + 1].squeeze(0).unsqueeze(1)
                    layer_kv.append((k_current, v_current))

                # 存储到缓存
                self._store_token_kv(seq.seq_id, [layer_kv], [slot])
        kv_store_time = time.perf_counter() - kv_store_start
        self.logger.info(f"KV storage time: {kv_store_time * 1000:.2f}ms")

        # 序列更新
        update_start = time.perf_counter()
        for i, seq in enumerate(batch):
            self._update_sequence(seq, next_tokens[i])
        update_time = time.perf_counter() - update_start
        self.logger.info(f"Sequence update time: {update_time * 1000:.2f}ms")

        total_time = time.perf_counter() - start_time
        self.logger.info(
            f"DECODE BATCH COMPLETED: "
            f"Total time: {total_time * 1000:.2f}ms ({len(batch)} seqs, "
            f"Avg: {total_time / len(batch) * 1000:.2f}ms/seq)\n"
        )

        return next_tokens


    def _update_sequence(self, seq: Sequence, next_token: int):
        """更新序列状态"""
        seq.update_state(next_token, None)

        # 流式输出回调
        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        self._invoke_stream_callback(seq.seq_id, next_token, token_text)

        if seq.is_finished():
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

        self.scheduler.running_sequences.clear()

        return results
