import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator
from queue import Queue
import time

from core.kv_store import  KVStore
from models.qwen_adapter import QwenModelAdapter
from .layer.layer import ModelLayerAdapter
from .scheduler import Scheduler
from .cache_manager import KVCacheManager, store_kvcache
from .sequence import Sequence
from .model_loader import load_model
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
            block_size = 256
            max_blocks = 32  # 40GB / (64 * 128 * 32 * 2 * 2) ≈ 640
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

        # 初始化KV存储管理器
        kv_store = KVStore(self.cache_manager, block_size)
        # 初始化模型层适配器
        self.layer_adapter = ModelLayerAdapter(self.model_config,
                                               device=device,
                                               num_heads=num_heads,
                                               kv_num_heads= num_key_value_heads,
                                               head_size=head_size,
                                               kv_store=kv_store)




        self.kv_store = kv_store

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

        """处理预填充批次 (极简版)"""
        # 1. 准备输入
        input_ids = [seq.get_next_input_ids() for seq in batch]
        input_tensor = self._pad_batch(input_ids, self.tokenizer.pad_token_id).to(self.device)

        # 2. 前向传播
        outputs = self.model(input_ids=input_tensor, use_cache=True)
        logits, past_kvs = outputs.logits, outputs.past_key_values

        # 3. 存储KV (按序列处理)
        for i, seq in enumerate(batch):
            num_tokens = len(input_ids[i])
            success, slot_mapping = self.cache_manager.alloc(seq.seq_id, num_tokens)
            if not success: continue
            model_type = getattr(self, 'model_type', 'default')
            for layer_idx, (k, v) in enumerate(past_kvs):
                # 直接提取当前序列的所有token KV (避免循环)
                if model_type == "qwen":
                    k_tensor = k[i, :num_tokens, :, :]  # [N, H, D]
                    v_tensor = v[i, :num_tokens, :, :]
                else:
                    k_tensor = k[i, :, :num_tokens, :].permute(1, 0, 2)  # [H, N, D]
                    v_tensor = v[i, :, :num_tokens, :].permute(1, 0, 2)
                k_cache, v_cache = self.cache_manager.get(layer_idx)
                store_kvcache(k_tensor, v_tensor, k_cache, v_cache,
                              torch.tensor(slot_mapping, dtype=torch.int32, device=k_tensor.device),
                              self.block_size)
        # 采样下一个token
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            next_tokens.append(next_token)
            self._update_sequence(seq, next_token)

        return next_tokens

    @torch.no_grad()
    def _process_decode_batch(self, batch: List[Sequence]):
        """处理解码批次，适配不同模型架构"""
        # 准备输入数据
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)
        token_positions = [[pos for pos in range(seq.current_position - 1)] for seq in batch]
        seq_ids = [seq.seq_id for seq in batch]

        hidden_states = self.embedding_layer(input_ids)

        # 逐层处理
        all_layer_kvs = []

        for i, seq in enumerate(batch):
            # 追加新的token
            self.cache_manager.append(seq.seq_id)
        context_lens = [seq.current_position for seq in batch]


        ## 追加新的token
        for layer_idx, layer in enumerate(self.model_layers):

            # 使用模型层适配器处理不同架构的层
            hidden_states, layer_kv = self.layer_adapter.process_layer(
                layer, hidden_states, self.cache_manager, seq_ids,
                context_lens, token_positions, layer_idx,
                [seq.current_position - 1 for seq in batch]
            )

            all_layer_kvs.append(layer_kv)


        # 最终层归一化 - 使用模型特定的归一化层
        hidden_states = self.norm_layer(hidden_states)
        logits = self.model.lm_head(hidden_states).float()

        # 采样下一个token
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            next_tokens.append(next_token)

        # 序列更新
        for i, seq in enumerate(batch):
            self._update_sequence(seq, next_tokens[i])


        return next_tokens


    def _update_sequence(self, seq: Sequence, next_token: int):
        """更新序列状态"""
        seq.update_state(next_token, None)

        # 流式输出回调
        token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        self._invoke_stream_callback(seq.seq_id, next_token, token_text)

        if seq.is_finished():
            self.scheduler.mark_finished(seq)
            self.cache_manager.free(seq.seq_id)
            self.logger.info(f"FINISHED: {seq.seq_id}")


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
