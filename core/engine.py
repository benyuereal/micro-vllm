
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Generator
from queue import Queue
import time

from models.qwen_adapter import QwenModelAdapter
from . import Scheduler
from .layer.layer import ModelLayerAdapter
from .cache_manager import KVCacheManager, store_kvcache
from .layer.optimized_qwen_layer import OptimizedQwenModelLayerAdapter
from .sequence import Sequence
from .model_loader import load_model
import logging

class InferenceEngine:


    # 设备配置 (可扩展)
    # 扩展DEVICE_CONFIGS (支持Qwen3 MoE)
    DEVICE_CONFIGS = {
        "mps": {"block_size": 16, "max_blocks": 512, "dtype": torch.float16},
        "cuda": {"block_size": 256, "max_blocks": 48, "dtype": torch.float16},  # 🔧 修复：强制使用float16以兼容CUDA内核
        "cpu": {"block_size": 16, "max_blocks": 128, "dtype": torch.float32},
        "cuda_moe": {"block_size": 256, "max_blocks": 48, "dtype": torch.bfloat16},  # ✅ Qwen3 MoE专用
    }

    def __init__(self, model_path: str, max_batch_size: int = 8, max_prefill_tokens: int = 2048):

        self.logger = self._init_logging()
        self.logger.info(f"Initializing InferenceEngine for {model_path}")

        # 1. 自动加载模型 (零拷贝)
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()


        # 2. 获取模型配置
        self.config = self.model.config
        self.model_type = self.config.model_type
        # 自动适配模型结构
        if self.model_type == "qwen":
            self.embedding_layer = self.model.transformer.wte
            self.norm_layer = self.model.transformer.ln_f
            self.model_layers = self.model.transformer.h
        elif self.model_type in ["qwen2", "qwen3"]:
            self.embedding_layer = self.model.model.embed_tokens
            self.norm_layer = self.model.model.norm
            self.model_layers = self.model.model.layers

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

        # 6. 初始化层适配器
        self.layer_adapter = OptimizedQwenModelLayerAdapter(
            model_config=self.config,
            device=self.device,
            num_heads=self.num_heads,
            head_size=self.head_size,kv_num_heads=self.kv_num_heads,
        )

        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens)
        self.adapter = QwenModelAdapter()

        self.eos_token_id = self.tokenizer.eos_token_id
        self.stream_callbacks = {}
        self.max_position = getattr(self.config, 'max_position_embeddings', 4096)

        self.logger.info(f"Engine initialized: layers={self.num_layers}, heads={self.num_heads}, "
                         f"block_size={self.block_size}, max_blocks={self.max_blocks}")

        # 8.其他配置


    def _init_logging(self):
        """初始化日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler("inference_perf.log"), logging.StreamHandler()]
        )
        return logging.getLogger("InferenceEngine")

    def _is_gptq_model(self):
        """检测是否为GPTQ模型"""

        return True

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

        # 关键修复：更健壮的GPTQ模型检测
        is_gptq_model = self._is_gptq_model()
        # 6. 初始化层适配器 - 传递GPTQ标识
        self.is_gptq_model = self._is_gptq_model()
        self.logger.info(f"Model type: {type(self.model)}, is_gptq: {is_gptq_model}")

        # 3. 获取模型当前的数据类型
        model_dtype = next(self.model.parameters()).dtype
        self.logger.info(f"Model dtype: {model_dtype}")

        # 🔧 修复：强制使用float16以兼容CUDA内核
        if device == "cuda":
            if is_gptq_model:
                # GPTQ模型强制使用float16以兼容CUDA内核
                dtype = torch.float16
                self.logger.info(f"GPTQ模型强制使用float16以兼容CUDA内核")
            else:
                # 非GPTQ模型也强制使用float16
                if model_dtype != torch.float16:
                    self.logger.info(f"Converting model from {model_dtype} to float16 for CUDA kernel compatibility")
                    self.model.to(torch.float16)
                dtype = torch.float16
                self.logger.info(f"非GPTQ模型也强制使用float16以兼容CUDA内核")
            
            # 🔧 额外修复：确保所有组件都使用float16
            if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
                if self.model.lm_head.weight.dtype != torch.float16:
                    self.logger.info(f"🔄 转换lm_head权重: {self.model.lm_head.weight.dtype} -> float16")
                    # 正确转换Parameter：保持Parameter的封装
                    self.model.lm_head.weight.data = self.model.lm_head.weight.data.to(torch.float16)
            
            # 🔧 确保embedding层也使用float16
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                if self.model.transformer.wte.weight.dtype != torch.float16:
                    self.logger.info(f"🔄 转换embedding权重: {self.model.transformer.wte.weight.dtype} -> float16")
                    self.model.transformer.wte.weight.data = self.model.transformer.wte.weight.data.to(torch.float16)
            
            # 🔧 确保norm层也使用float16
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
                for param in self.model.transformer.ln_f.parameters():
                    if param.dtype != torch.float16:
                        self.logger.info(f"🔄 转换norm层参数: {param.dtype} -> float16")
                        param.data = param.data.to(torch.float16)

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
        """执行一个推理step，返回是否有处理任何请求"""
        batch, batch_type = self.scheduler.get_next_batch()
        if not batch:
            return False

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
        # 🔧 修复：确保输入数据类型正确
        if input_tensor.dtype != torch.long:
            input_tensor = input_tensor.long()
        
        # 确保模型在正确的数据类型下运行
        with torch.cuda.amp.autocast(enabled=False):
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
        """处理解码批次，适配不同模型架构 - 添加性能埋点"""
        # 初始化阶段耗时统计
        stage_times = {
            "input_prep": 0,  # 输入准备
            "embedding": 0,  # 嵌入层处理
            "layers": 0,  # 所有层处理
            "norm_logits": 0,  # 归一化和logits计算
            "sampling": 0,  # token采样
            "seq_update": 0  # 序列更新
        }

        # 阶段1: 准备输入数据
        start_time = time.time()
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=self.device)
        token_positions = [[pos for pos in range(seq.current_position - 1)] for seq in batch]
        seq_ids = [seq.seq_id for seq in batch]
        stage_times["input_prep"] = time.time() - start_time

        # 阶段2: 嵌入层处理
        start_time = time.time()
        hidden_states = self.embedding_layer(input_ids)
        stage_times["embedding"] = time.time() - start_time

        # 阶段3: 缓存更新
        start_time = time.time()
        for i, seq in enumerate(batch):
            self.cache_manager.append(seq.seq_id)
        context_lens = [seq.current_position for seq in batch]
        self.cache_manager.cache_batch_data(seq_ids, context_lens)
        cache_time = time.time() - start_time

        # 阶段4: 所有层处理（合并计时）
        start_time = time.time()
        all_layer_kvs = []
        for layer_idx, layer in enumerate(self.model_layers):
            hidden_states, layer_kv = self.layer_adapter.process_layer(
                layer, hidden_states, self.cache_manager, seq_ids,
                context_lens, token_positions, layer_idx,
                [seq.current_position - 1 for seq in batch]
            )
            all_layer_kvs.append(layer_kv)
        stage_times["layers"] = time.time() - start_time

        # 阶段5: 归一化和logits计算
        start_time = time.time()
        hidden_states = self.norm_layer(hidden_states)
        logits = self.model.lm_head(hidden_states).float()
        stage_times["norm_logits"] = time.time() - start_time

        # 阶段6: 采样下一个token
        start_time = time.time()
        next_tokens = []
        for i, seq in enumerate(batch):
            next_token = self._sample_next_token(logits[i, -1, :], seq.temperature, seq.top_p)
            next_tokens.append(next_token)
        stage_times["sampling"] = time.time() - start_time

        # 阶段7: 序列更新
        start_time = time.time()
        for i, seq in enumerate(batch):
            self._update_sequence(seq, next_tokens[i])
        stage_times["seq_update"] = time.time() - start_time

        # 计算总耗时和吞吐量

        # 详细耗时分布（调试级别）
        self.logger.info(
            f"STAGE BREAKDOWN | "
            f"Input: {stage_times['input_prep'] * 1000:.1f}ms | "
            f"Embed: {stage_times['embedding'] * 1000:.1f}ms | "
            f"Layers: {stage_times['layers'] * 1000:.1f}ms | "
            f"Norm: {stage_times['norm_logits'] * 1000:.1f}ms | "
            f"Sample: {stage_times['sampling'] * 1000:.1f}ms | "
            f"Update: {stage_times['seq_update'] * 1000:.1f}ms"
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
