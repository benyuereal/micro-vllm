"""
===================================================================
InferenceEngine - Micro-vLLM 推理引擎
===================================================================
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
from core.parallel_config import get_rank, setup, get_world_size

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("inference_perf.log"), logging.StreamHandler()]
)

logger = logging.getLogger("InferenceEngine")

class InferenceEngine:
    """
    📌 推理引擎 - vLLM核心组件

    🔍 设计哲学:
        1. 极简接口: 自动加载模型
        2. 自动适配: 根据模型类型自动选择最佳配置
        3. 零拷贝: 直接操作模型参数

    🧪 典型用法:
        engine = InferenceEngine(model_path="Qwen/Qwen2-0.5B", max_batch_size=8)
        output = engine.generate(input_ids, max_new_tokens=100)
    """

    # 设备配置
    DEVICE_CONFIGS = {
        "mps": {"block_size": 16, "max_blocks": 512, "dtype": torch.float16},
        "cuda": {"block_size": 256, "max_blocks": 65, "dtype": None},  # 自动选择bfloat16/float16
        "cpu": {"block_size": 16, "max_blocks": 128, "dtype": torch.float32},
        "cuda_moe": {"block_size": 256, "max_blocks": 48, "dtype": torch.bfloat16},
    }

    def __init__(self, model_path: str, max_batch_size: int = 32, max_prefill_tokens: int = 2048):
        """
        📌 初始化推理引擎

        🔍 参数:
            - model_path: 模型路径
            - max_batch_size: 最大批次大小
            - max_prefill_tokens: 最大预填充token数
        """

        # tp配置
        setup()
        rank = get_rank()
        if torch.cuda.is_available():
           torch.cuda.set_device(rank)
        
        logger.info(f"Initializing InferenceEngine for {model_path}")

        # 1. 自动加载模型
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = load_model(model_path, device=device)
        self.model.eval()
        logger.info(f"Rank {rank}: ln_f weight device = {self.model.transformer.ln_f.weight.device}")
        
        # 2. 获取模型配置
        self.config = self.model.config
        self.model_type = self.config.model_type
        self.embedding_layer = self.model.transformer.wte
        self.norm_layer = self.model.transformer.ln_f
        self.model_layers = self.model.transformer.h
        
        logger.info(f"Detected model type: {self.model_type}")

        # 3. 自动配置参数
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        
        # ✅ 修复：从模型配置获取 head_size (支持Qwen3)
        if hasattr(self.config, 'head_dim'):
            self.head_size = self.config.head_dim
        else:
            self.head_size = self.config.hidden_size // self.num_heads
        
        world_size = get_world_size()
        # 维度校验
        assert self.num_heads % world_size == 0, f"num_heads {self.num_heads} must be divisible by world_size {world_size}"
        assert self.kv_num_heads % world_size == 0, f"kv_num_heads {self.kv_num_heads} must be divisible by world_size {world_size}"
        # 执行切分
        self.num_heads = self.num_heads // world_size
        self.kv_num_heads = self.kv_num_heads // world_size

        logger.info(f"Rank {rank}: head_size = {self.config.hidden_size}, num_heads = {self.num_heads}, kv_num_heads = {self.kv_num_heads}, head_size = {self.head_size}")
        
        # 4. 自动检测设备和精度
        self.device, self.dtype, self.block_size, self.max_blocks = self._auto_configure()
        logger.info(f"Using device={self.device}, dtype={self.dtype}")

        # 5. 初始化核心模块
        self.cache_manager = KVCacheManager(
            n_blocks=self.max_blocks,
            block_size=self.block_size,
            n_layers=self.num_layers,
            n_heads=self.kv_num_heads,
            head_size=self.head_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # 6. 初始化模型层Graph运行器
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
        self.pad_token_id = 0
        self.stream_callbacks = {}
        self.max_position = getattr(self.config, 'max_position_embeddings', 4096)

        logger.info(f"Engine initialized: layers={self.num_layers}, heads={self.num_heads}, "
                         f"block_size={self.block_size}, max_blocks={self.max_blocks}")
        self.sampler = Sampler()
        
        # 7. CUDA Graph 配置
        if self.device == "cuda":
            logger.info("Starting CUDA Graphs capture...")
            self.graph_runner.capture(
                self.cache_manager,
                batch_sizes=[1, 2, 4, 8, 16, 32]
            )
            logger.info("CUDA Graphs capture completed")

        

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

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        """获取下一个批次"""
        return self.scheduler.get_next_batch()

            
    @torch.no_grad()
    def step(self, batch: List[Sequence], batch_type: str) -> bool:
        """执行推理step：接受batch进行推理"""
        if not batch:
            return False
            
        if batch_type == "prefill":
            self._process_prefill_batch(batch)
        elif batch_type == "decode":
            self._process_decode_batch(batch)
            
        return True

    @torch.no_grad()
    def _process_prefill_batch(self, batch: List[Sequence]):
        """处理预填充批次"""
        start_time = time.time()
        batch_size = len(batch)
        if not batch_size: return []

        # 1. 准备输入数据
        prep_start = time.time()
        device = self.device
        
        # 定长分桶，直接 stack
        input_ids_list = [seq.get_next_input_ids() for seq in batch]
        input_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        
        temperatures = torch.tensor([seq.temperature for seq in batch], device=device)
        top_ps = torch.tensor([seq.top_p for seq in batch], device=device)
        
        seq_lens = [len(ids) for ids in input_ids_list]
        seq_len = seq_lens[0]

        # 2. KV Cache 管理
        seq_ids = []
        for i, seq in enumerate(batch):
            success, _ = self.cache_manager.alloc(seq.seq_id, seq_len)
            if not success:
                raise RuntimeError(f"OOM allocating cache for seq {seq.seq_id}")
            seq_ids.append(seq.seq_id)
            
        self.cache_manager.cache_batch_data(seq_ids, seq_lens)
        
        prep_time = time.time() - prep_start

        # 3. 推理
        gpu_start = time.time()
        logits = self.graph_runner.prefill(
            input_tensor, 
            self.cache_manager, 
            batch_size
        )
        gpu_time = time.time() - gpu_start

        # 4. 采样
        sample_start = time.time()
        last_logits = logits[:, -1, :]
        next_tokens = self.sampler(last_logits, temperatures, top_ps, 1000).tolist()
        sample_time = time.time() - sample_start

        # 5. 更新状态
        update_start = time.time()
        for i, seq in enumerate(batch):
            self._update_sequence(seq, next_tokens[i])
        update_time = time.time() - update_start

        # 6. 日志
        total_time = time.time() - start_time
        logger.info(
            f"📝 预填充 (Prefill): "
            f"Prep {prep_time*1000:.2f}ms, GPU {gpu_time*1000:.2f}ms, "
            f"Sample {sample_time*1000:.2f}ms, Update {update_time*1000:.2f}ms, "
            f"Total {total_time*1000:.2f}ms | Batch{batch_size}, Len{seq_len}"
        )

        return next_tokens

    @torch.no_grad()
    def _process_decode_batch(self, batch: List[Sequence]):
        """处理解码批次"""
        start_time = time.time()
        batch_size = len(batch)
        if not batch_size: return []

        # 1. 构建 mask 和 indices
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
        
        logits = self.graph_runner.forward(
            input_ids, 
            self.cache_manager, 
            batch_size
        )
        
        # 5. 采样
        sample_start = time.time()
        
        next_tokens = self.sampler(logits, temperatures, top_ps, 1000).tolist()
        sample_time = time.time() - sample_start
        
        gpu_time = time.time() - gpu_start
        
        # 6. 更新状态
        update_start = time.time()
        for i in indices:
            self._update_sequence(batch[i], next_tokens[i])
        update_time = time.time() - update_start
        
        # 7. 日志
        total_time = time.time() - start_time
        if indices and batch[indices[0]].current_position % 50 == 0:
            logger.info(
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
            logger.info(f"FINISHED: {seq.seq_id}")


    def _pad_batch(self, sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
        """填充批次"""
        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    

    def generate(self, prompts: List[str], max_tokens: int = 100) -> Dict[Sequence, str]:
        """批量生成"""
        # 添加所有请求
        seq_ids = [self.add_request(prompt, max_tokens) for prompt in prompts]
        seq_map = {seq_id: prompt for seq_id, prompt in zip(seq_ids, prompts)}

        # 执行推理直到完成
        for _ in range(max_tokens):
            batch , batch_type = self.get_next_batch()
            if not self.step(batch, batch_type) and not self.scheduler.running_sequences:
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
