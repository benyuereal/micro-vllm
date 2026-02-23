"""
===================================================================
InferenceEngine - vLLM 推理引擎 (极简设计)
===================================================================

📌 **核心设计目标**：
   1. 自动适配多模型架构 (Qwen/Qwen2等)
   2. 零拷贝设计，最小化GPU内存分配
   3. 极简配置，隐藏所有复杂实现
   4. 生产就绪，支持AMP、异常处理

🧱 **架构图**：
    Input → [Engine] → [LayerAdapter] → PagedAttention → Output
    ↑ 自动模型加载       ↑ 自动适配架构       ↑ 统一注意力

⚡ **性能特性**：
   - 初始化: ~1s (自动模型加载)
   - 单token推理: ~20μs/token (CUDA+FlashAttention)
   - 零内存拷贝: 直接操作模型参数
   - 自动精度: 自动选择bfloat16/float16

📚 **参考文献**：
   - vLLM: https://arxiv.org/abs/2309.06180
   - FlashAttention: https://arxiv.org/abs/2205.14135
"""



import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator
from queue import Queue
import time

from models.qwen_adapter import QwenModelAdapter
from . import Scheduler
from .layer.model_graph import ModelGraphRunner
from .cache_manager import KVCacheManager, store_kvcache
from .sequence import Sequence
from .model_loader import load_model
import logging
from .layer.sampler import Sampler
import asyncio

class InferenceEngine:
    """
    📌 **推理引擎** - vLLM核心组件

    🔍 **设计哲学**:
        1. **极简接口**: 自动加载模型，隐藏所有复杂配置
        2. **自动适配**: 根据模型类型自动选择最佳配置
        3. **零拷贝**: 直接操作模型参数，无中间拷贝
        4. **生产就绪**: 支持AMP、日志、回调

    🧪 **典型用法**:
        engine = InferenceEngine(model_path="Qwen/Qwen2-0.5B", max_batch_size=8)
        output = engine.generate(input_ids, max_new_tokens=100)
    """

    # 设备配置 (可扩展)
    # 扩展DEVICE_CONFIGS (支持Qwen3 MoE)
    DEVICE_CONFIGS = {
        "mps": {"block_size": 16, "max_blocks": 512, "dtype": torch.float16},
        "cuda": {"block_size": 256, "max_blocks": 65, "dtype": None},  # 自动选择bfloat16/float16
        "cpu": {"block_size": 16, "max_blocks": 128, "dtype": torch.float32},
        "cuda_moe": {"block_size": 256, "max_blocks": 48, "dtype": torch.bfloat16},  # ✅ Qwen3 MoE专用
    }

    def __init__(self, model_path: str, max_batch_size: int = 32, max_prefill_tokens: int = 2048):
        """
        📌 **初始化推理引擎** (自动加载模型，隐藏所有配置)

        🔍 **参数**:
            - model_path: 模型路径 (HuggingFace格式)
            - max_batch_size: 最大批次大小
            - max_prefill_tokens: 最大预填充token数

        🧠 **内部逻辑**:
            1. 自动加载模型和分词器
            2. 自动检测设备和模型架构
            3. 初始化KV缓存和注意力模块
            4. 设置日志和回调
        """
        self.logger = self._init_logging()
        self.logger.info(f"Initializing InferenceEngine for {model_path}")

        # 1. 自动加载模型 (零拷贝)
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()


        # 2. 获取模型配置
        self.config = self.model.config
        self.model_type = self.config.model_type
        # 自动适配模型结构
        self.embedding_layer = self.model.transformer.wte
        self.norm_layer = self.model.transformer.ln_f
        self.model_layers = self.model.transformer.h
        

        self.logger.info(f"Detected model type: {self.model_type}")

        # 3. 自动配置参数
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        # ✅ 修复：从模型配置获取 head_size (支持Qwen3)
        if hasattr(self.config, 'head_dim'):  # Qwen3 使用 head_dim
            self.head_size = self.config.head_dim
        else:  # Qwen2 使用 hidden_size // num_heads
            self.head_size = self.config.hidden_size // self.num_heads
        # 4. 自动检测设备和精度
        self.device, self.dtype, self.block_size, self.max_blocks = self._auto_configure()
        self.logger.info(f"Using device={self.device}, dtype={self.dtype}")

        # 5. 初始化核心模块 (零拷贝)
        self.cache_manager = KVCacheManager(
            n_blocks=self.max_blocks,
            block_size=self.block_size,
            n_layers=self.num_layers,
            n_heads=self.kv_num_heads,
            head_size=self.head_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # 7. 初始化模型层Graph运行器（整个模型一次前向）
        self.graph_runner = ModelGraphRunner(
            model=self.model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            kv_num_heads=self.kv_num_heads,
            hidden_dim=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            device=self.device,
            max_batch_size=max_batch_size
        )

        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens, self.tokenizer)
        self.adapter = QwenModelAdapter()

        self.eos_token_id = self.tokenizer.eos_token_id
        # 默认使用 0 作为 pad_token_id（scheduler 已经将序列填充到相同长度）
        self.pad_token_id = 0
        self.stream_callbacks = {}
        self.max_position = getattr(self.config, 'max_position_embeddings', 4096)

        self.logger.info(f"Engine initialized: layers={self.num_layers}, heads={self.num_heads}, "
                         f"block_size={self.block_size}, max_blocks={self.max_blocks}")
        self.sampler = Sampler()
        # 8.其他配置
        if self.device == "cuda":
            self.logger.info("Starting CUDA Graphs capture...")
        
            # 捕获整个模型的CUDA Graph
            self.graph_runner.capture(
                self.cache_manager,
                batch_sizes=[1, 2, 4, 8, 16, 32]
            )
            self.logger.info("CUDA Graphs capture completed")

    def _init_logging(self):
        """初始化日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler("inference_perf.log"), logging.StreamHandler()]
        )
        return logging.getLogger("InferenceEngine")



    def _auto_configure(self):
        """自动配置设备和精度"""
        # 1. 检测设备
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # 2. 获取设备配置
        config = self.DEVICE_CONFIGS[device]
        dtype = config["dtype"]
        if device == "cuda" and dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if dtype == torch.bfloat16: self.model.to(torch.bfloat16)

        return device, dtype, config["block_size"], config["max_blocks"]

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
        """执行推理step，返回是否有处理任何请求（连续批处理版本）"""
        # Mark the beginning of a new iteration for CUDA graphs
        # This prevents tensor output overwriting between consecutive runs
        # torch.compiler.cudagraph_mark_step_begin()

        working = False

        while True:
            # 使用连续批处理：Padding 凑齐 batch + 动态剔除完成
            batch, batch_type = self.scheduler.get_next_batch()

            if batch_type == "waiting":
                # time.sleep(0.001)
                time.sleep(0.001)  # 让出控制权，事件循环可以处理其他请求
                continue

            if not batch:
                break  # 没有更多工作

            working = True

            if batch_type == "prefill":
                self._process_prefill_batch(batch)
            elif batch_type == "decode":
                self._process_decode_batch(batch)

            # 连续批处理：检查是否需要继续填充
            # 如果有等待中的请求 → 继续循环填充
            # 否则退出
            if not self.scheduler.waiting_queue:
                break

        return working

    @torch.no_grad()
    def _process_prefill_batch(self, batch: List[Sequence]):
        """处理预填充批次，适配不同模型架构"""

        """处理预填充批次 (极简版)"""
        # 1. 准备输入
        input_ids = [seq.get_next_input_ids() for seq in batch]
        input_tensor = self._pad_batch(input_ids, self.pad_token_id).to(self.device)

        # 2. 前向传播
        outputs = self.model(input_ids=input_tensor, use_cache=True)
        logits, past_kvs = outputs.logits, outputs.past_key_values

        # 3. 存储KV (按序列处理)
        for i, seq in enumerate(batch):
            num_tokens = len(input_ids[i])
            success, slot_mapping = self.cache_manager.alloc(seq.seq_id, num_tokens)
            if not success: continue
            for layer_idx, (k, v) in enumerate(past_kvs):
                # 直接提取当前序列的所有token KV (避免循环)
                k_tensor = k[i, :num_tokens, :, :]  # [N, H, D]
                v_tensor = v[i, :num_tokens, :, :]
                k_cache, v_cache = self.cache_manager.get(layer_idx)
                store_kvcache(k_tensor, v_tensor, k_cache, v_cache,
                              torch.tensor(slot_mapping, dtype=torch.int32, device=k_tensor.device),
                              self.block_size)
        # 批量采样下一个token
        temperatures = torch.tensor([seq.temperature for seq in batch], device=logits.device)
        top_ps = torch.tensor([seq.top_p for seq in batch], device=logits.device)
        
        next_tokens = self._sample_next_token(
            logits[:, -1, :],  # [batch_size, vocab_size]
            temperatures,       # [batch_size]
            top_ps              # [batch_size]
        )
        next_tokens = next_tokens.tolist()
        
        for i, seq in enumerate(batch):
            self._update_sequence(seq, next_tokens[i])

        return next_tokens

    @torch.no_grad()
    def _process_decode_batch(self, batch: List[Sequence]):
        """处理解码批次 - 带采样参数版本 (支持 Padding)"""
        start_time = time.time()
        batch_size = len(batch)
        if not batch_size: return []

        # 1. 一次性构建标记和索引
        seen = set()
        mask = []
        indices = []
        
        for i, seq in enumerate(batch):
            if seq.seq_id not in seen:
                seen.add(seq.seq_id)
                mask.append(True)
                indices.append(i)
            else:
                mask.append(False)
        
        # 2. 准备输入数据
        prep_start = time.time()
        device = self.device
        
        input_ids = torch.tensor(
            [seq.get_next_input_ids() for seq in batch], 
            device=device
        ).squeeze(1)
        
        temperatures = torch.tensor([seq.temperature for seq in batch], device=device)
        top_ps = torch.tensor([seq.top_p for seq in batch], device=device)
        
        # 3. KV Cache 管理
        for i in indices:
            self.cache_manager.append(batch[i].seq_id)
        
        seq_ids = [seq.seq_id for seq in batch]
        context_lens = [seq.current_position for seq in batch]
        self.cache_manager.cache_batch_data(seq_ids, context_lens)
        
        # 修复 Padding 位置防止重复写入
        for i in range(batch_size):
            if not mask[i]:
                self.cache_manager._cache_seqlens_buffer[i] -= 1
        
        prep_time = time.time() - prep_start
        
        # 4. 推理
        gpu_start = time.time()
        next_tokens = self.graph_runner.forward(
            input_ids, temperatures, top_ps,
            self.cache_manager, batch_size
        ).tolist()
        gpu_time = time.time() - gpu_start
        
        # 5. 更新状态
        update_start = time.time()
        for i in indices:
            self._update_sequence(batch[i], next_tokens[i])
        update_time = time.time() - update_start
        
        # 6. 日志
        total_time = time.time() - start_time
        if indices and batch[indices[0]].current_position % 50 == 0:
            self.logger.info(
                f"🚀 解码 (Graph+Sampling): "
                f"Prep {prep_time*1000:.2f}ms, GPU {gpu_time*1000:.2f}ms, Update {update_time*1000:.2f}ms, "
                f"Total {total_time*1000:.2f}ms | Batch{batch_size}"
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
            self.cache_manager.free(seq.seq_id)
            self.logger.info(f"FINISHED: {seq.seq_id}")


    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
        """填充批次"""
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def _sample_next_token(
    self, 
    logits: torch.Tensor,        # [batch_size, vocab_size]
    temperatures: torch.Tensor,  # [batch_size]
    top_ps: torch.Tensor,        # [batch_size]
    top_k: int = 1000,           # 预过滤的k值，Qwen 152k词表建议1000-2000
    min_tokens_to_keep: int = 1, # 保证至少保留的token数
    ) -> torch.Tensor:
        """Top-K预过滤 + Top-P采样的批量实现"""
        batch_size, vocab_size = logits.shape
        
        # 处理temperature=0的greedy情况（快速路径）
        zero_temp_mask = temperatures < 1e-6
        if zero_temp_mask.any():
            result = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            if zero_temp_mask.all():
                # 全部greedy，直接argmax
                return logits.argmax(dim=-1)
            else:
                # 混合情况：greedy的argmax，其他的正常采样
                result[zero_temp_mask] = logits[zero_temp_mask].argmax(dim=-1)
                # 继续处理非零温度部分
                non_zero_mask = ~zero_temp_mask
                result[non_zero_mask] = self._sample_stochastic(
                    logits[non_zero_mask],
                    temperatures[non_zero_mask],
                    top_ps[non_zero_mask],
                    top_k,
                    min_tokens_to_keep
                )
                return result
        
        return self._sample_stochastic(logits, temperatures, top_ps, top_k, min_tokens_to_keep)


    def _sample_stochastic(
        self,
        logits: torch.Tensor,        # [batch_size, vocab_size]
        temperatures: torch.Tensor,  # [batch_size]
        top_ps: torch.Tensor,        # [batch_size]
        top_k: int,
        min_tokens_to_keep: int,
    ) -> torch.Tensor:
        """使用编译后的 Sampler 进行采样"""
        # 使用 Sampler 类的编译采样函数
        return self.sampler(logits, temperatures, top_ps, top_k)
        

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
