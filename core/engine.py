import torch
import time
import logging
import atexit
from typing import List, Dict, Tuple
from dataclasses import dataclass

from . import Scheduler
from .layer.model_graph import ModelGraphRunner
from .cache_manager import KVCacheManager
from .sequence import Sequence
from .model_loader import load_model
from .layer.sampler import Sampler
from core.parallel_config import get_rank, setup, get_world_size, rank0
from core.inference_context import BatchInferenceContext
from torch.profiler import profile, ProfilerActivity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("inference_perf.log"), logging.StreamHandler()]
)
logger = logging.getLogger("InferenceEngine")

@dataclass
class InferenceStats:
    prep_time: float = 0.0
    gpu_time: float = 0.0
    sample_time: float = 0.0
    total_time: float = 0.0

class InferenceEngine:
    """
    Micro-vLLM 推理引擎核心类。
    负责模型加载、调度、KVCache管理及执行推理。
    """
    
    # 预设配置 (简化为仅保留关键逻辑，硬编码CUDA最优实践)
    DEFAULT_BLOCK_SIZE = 256
    DEFAULT_MAX_BLOCKS = 65

    def __init__(self, model_path: str, max_batch_size: int = 32, max_prefill_tokens: int = 2048):
        self._init_distributed()
        self._init_model(model_path)
        self._init_config()
        
        # 核心组件初始化
        self.device, self.dtype = self._auto_configure()
        
        self.cache_manager = KVCacheManager(
            n_blocks=self.DEFAULT_MAX_BLOCKS, block_size=self.DEFAULT_BLOCK_SIZE,
            n_layers=self.num_layers, n_heads=self.kv_num_heads, head_size=self.head_size,
            dtype=self.dtype, device=self.device
        )
        self.graph_runner = ModelGraphRunner(
            model=self.model, num_layers=self.num_layers, num_heads=self.num_heads,
            head_size=self.head_size, kv_num_heads=self.kv_num_heads,
            hidden_dim=self.config.hidden_size, intermediate_size=self.intermediate_size,
            device=self.device, max_batch_size=max_batch_size, dtype=self.dtype
        )
        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens, self.tokenizer)
        self.sampler = Sampler()
        
        # 状态
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stream_callbacks = {}
        self.max_position = getattr(self.config, 'max_position_embeddings', 4096)
        
        # 捕获 CUDA Graph
        if self.device == "cuda":
            logger.info("Capturing CUDA Graphs...")
            self.graph_runner.capture(self.cache_manager, batch_sizes=[1, 2, 4, 8, 16, 32])
            logger.info("CUDA Graphs captured.")
            
        # 注册退出钩子
        atexit.register(self.shutdown)

    def _init_distributed(self):
        setup()
        self.rank = get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
        self.device_str = f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"

    def _init_model(self, model_path: str):
        logger.info(f"Loading model {model_path} on rank {self.rank}")
        self.model, self.tokenizer = load_model(model_path, device=self.device_str)
        self.model.eval()
        self.config = self.model.config

    def _init_config(self):
        # 提取模型配置
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.kv_num_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_size = getattr(self.config, 'head_dim', self.config.hidden_size // self.num_heads)
        
        # 张量并行切分
        world_size = get_world_size()
        assert self.num_heads % world_size == 0 and self.kv_num_heads % world_size == 0
        
        self.num_heads //= world_size
        self.kv_num_heads //= world_size
        self.intermediate_size = self.config.intermediate_size // world_size

    def _auto_configure(self) -> Tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if dtype == torch.bfloat16:
                self.model.to(torch.bfloat16)
            return "cuda", dtype
        return "cpu", torch.float32

    def shutdown(self):
        """
        清理资源，主要用于优雅退出分布式环境。
        """
        try:
            if torch.distributed.is_initialized():
                logger.info(f"Rank {self.rank}: Shutting down distributed process group...")
                torch.distributed.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")

    # -------------------------------------------------------------------------
    # 公共接口 (Public API)
    # -------------------------------------------------------------------------

    def add_request(self, prompt: str, max_tokens: int = 128, 
                    temperature: float = 0.7, top_p: float = 0.9) -> int:
        seq_id = hash(prompt + str(time.time())) % (2 ** 32)
        seq = Sequence(seq_id, prompt, self.tokenizer, max_tokens)
        seq.temperature = temperature
        seq.top_p = top_p
        self.scheduler.add_request(seq)
        return seq_id

    def register_stream_callback(self, seq_id: int, callback):
        self.stream_callbacks[seq_id] = callback

    def unregister_stream_callback(self, seq_id: int):
        self.stream_callbacks.pop(seq_id, None)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        return self.scheduler.get_next_batch()

    @torch.no_grad()
    def step(self, ctx: BatchInferenceContext) -> bool:
        if not ctx.sequences: return False
        
        if ctx.batch_type == "prefill":
            self._prefill(ctx.sequences)
        else:
            # 仅包裹 Decode 阶段
            self._decode(ctx.sequences)
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True,
            #     profile_memory=True
            # ) as prof:
            #     self._decode(ctx.sequences)
            # # 可选：如果需要保存 profiler 结果，取消下面注释
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            # prof.export_chrome_trace(f"decode_trace_{time.time()}.json")
        
        return True

    def generate(self, prompts: List[str], max_tokens: int = 100) -> Dict[str, str]:
        seq_ids = [self.add_request(p, max_tokens) for p in prompts]
        seq_map = {sid: p for sid, p in zip(seq_ids, prompts)}
        
        # 简易事件循环
        for _ in range(max_tokens * len(prompts)): # 安全上限
            batch, batch_type = self.get_next_batch()
            if not batch and not self.scheduler.running_sequences: break
            
            ctx = BatchInferenceContext(len(batch), batch_type, batch)
            self.step(ctx)

        # 结果收集
        results = {}
        for seq, out_ids in self.scheduler.get_finished_results():
            try:
                results[seq_map[seq.seq_id]] = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            except:
                results[seq_map[seq.seq_id]] = f"[Error]"
        
        self.scheduler.running_sequences.clear()
        return results

    # -------------------------------------------------------------------------
    # 内部逻辑 (Internal Logic)
    # -------------------------------------------------------------------------

    def _prefill(self, batch: List[Sequence]):
        stats = InferenceStats(total_time=time.time())
        
        # 1. 准备
        stats.prep_time = time.time()
        device = self.device
        
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], dtype=torch.long, device=device)
        temps = torch.tensor([seq.temperature for seq in batch], device=device)
        topp = torch.tensor([seq.top_p for seq in batch], device=device)
        seq_lens = [len(ids) for ids in input_ids]
        
        # 2. KV Cache 分配
        for seq in batch:
            ok, _ = self.cache_manager.alloc(seq.seq_id, seq_lens[0])
            if not ok: raise RuntimeError("OOM")
        
        self.cache_manager.cache_batch_data([s.seq_id for s in batch], seq_lens)
        stats.prep_time = time.time() - stats.prep_time

        # 3. 推理
        stats.gpu_time = time.time()
        logits = self.graph_runner.prefill(input_ids, self.cache_manager, len(batch))
        stats.gpu_time = time.time() - stats.gpu_time

        # 4. 采样
        if rank0():
            stats.sample_time = time.time()
            next_tokens = self.sampler(logits[:, -1, :], temps, topp, 1000).tolist()
            for i, seq in enumerate(batch):
                seq._next_token = next_tokens[i]
            stats.sample_time = time.time() - stats.sample_time
            
            stats.total_time = time.time() - stats.total_time
            logger.info(f"Prefill: Prep {stats.prep_time*1000:.1f}ms, GPU {stats.gpu_time*1000:.1f}ms, Total {stats.total_time*1000:.1f}ms | Batch {len(batch)}")

    def _decode(self, batch: List[Sequence]):
        stats = InferenceStats(total_time=time.time())
        device = self.device
        batch_size = len(batch)

        # 1. 准备与去重
        stats.prep_time = time.time()
        seen = set()
        mask = []
        for seq in batch:
            mask.append(seq.seq_id not in seen)
            seen.add(seq.seq_id)

        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], device=device).squeeze(1)
        temps = torch.tensor([seq.temperature for seq in batch], device=device)
        topp = torch.tensor([seq.top_p for seq in batch], device=device)

        # 2. KV Cache Append
        for i, seq in enumerate(batch):
            if mask[i]: self.cache_manager.append(seq.seq_id)
        
        self.cache_manager.cache_batch_data([s.seq_id for s in batch], [s.current_position for s in batch])
        
        # 回滚 padding 位置
        for i in range(batch_size):
            if not mask[i]: self.cache_manager._cache_seqlens_buffer[i] -= 1
        
        stats.prep_time = time.time() - stats.prep_time

        # 3. 推理
        stats.gpu_time = time.time()
        logits = self.graph_runner.forward(input_ids, self.cache_manager, batch_size)
        stats.gpu_time = time.time() - stats.gpu_time

        # 4. 采样
        if rank0():
            stats.sample_time = time.time()
            next_tokens = self.sampler(logits, temps, topp, 1000).tolist()
            for i, seq in enumerate(batch):
                seq._next_token = next_tokens[i]
            stats.sample_time = time.time() - stats.sample_time

            stats.total_time = time.time() - stats.total_time
            if batch and batch[0].current_position % 50 == 0:
                logger.info(f"Decode: Prep {stats.prep_time*1000:.1f}ms, GPU {stats.gpu_time*1000:.1f}ms, Total {stats.total_time*1000:.1f}ms | Batch {batch_size}")

    def update_sequences(self, sequences: List[Sequence]):
        for seq in sequences:
            if seq._next_token is None: continue
            
            seq.update_state(seq._next_token, None)
            
            if rank0():
                # 流式输出
                txt = self.tokenizer.decode([seq._next_token], skip_special_tokens=True)
                if cb := self.stream_callbacks.get(seq.seq_id):
                    try: cb(seq._next_token, txt)
                    except: pass
            
            if seq.is_finished():
                self.cache_manager.free(seq.seq_id)
                if rank0():
                    self.scheduler.mark_finished(seq)