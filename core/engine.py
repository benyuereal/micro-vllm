import torch
import time
import asyncio
import logging
import atexit
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from . import Scheduler
from .layer.model_graph import ModelGraphRunner
from .layer.model_prefill import ModelPrefillRunner
from .cache_manager import KVCacheManager
from .sequence import Sequence
from .model_loader import load_model
from .layer.sampler import Sampler
from .context import DecodeContext

from core.parallel_config import get_rank, setup, get_world_size, rank0
from core.inference_context import BatchInferenceContext
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("inference_perf.log"), logging.StreamHandler()]
)
logger = logging.getLogger("InferenceEngine")


@dataclass
class StreamEvent:
    """一步 decode 产生的待推送数据：token 列表 + 对应的流式回调。"""
    tokens:    List[int]
    callbacks: list


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

    def __init__(self, model_path: str, max_batch_size: int = 512, max_prefill_tokens: int = 8192,
                 max_context_length: int = 1024):
        self._init_distributed()
        self._init_model(model_path)
        self._init_config()

        # 核心组件初始化
        self.device, self.dtype = self._auto_configure()

        # 序列长度上限：可配（构造参数 max_context_length）。DeepSeek MLA decode kernel
        # 把 max_len 进静态 shape（block_table 列数 / cos_k 行数），故 DeepSeek 固定 1024
        #（见 model_graph._deepseek_fixed_maxlen）；Qwen3 用 flash_attn_with_kvcache，
        # seq_len 不进 kernel shape（cache_seqlens per-seq 截断），可放开到任意配置值。
        # 架构侧硬上限（adapter.context_length_limit）优先：DeepSeek 钳到 1024，
        # Qwen3 返回 None（无架构限制，取配置值）。
        arch_limit = self.adapter.context_length_limit(self.config)
        self.max_position = min(max_context_length, arch_limit) if arch_limit else max_context_length
        max_seq_blocks = (self.max_position + self.DEFAULT_BLOCK_SIZE - 1) // self.DEFAULT_BLOCK_SIZE
        # n_blocks 按显存预算推导，与 max_batch_size 解耦（vLLM V2 思路：固定停车位总数
        # 由可用显存决定，而非 max_batch × max_position 全量预分配——后者在 max_batch=512
        # 时需 75GB 直接 OOM）。预算 = 剩余显存 - 余量（权重/graph buffer/activation）。
        # 单 block 全层 K+V 显存 = 2(K+V) × block_size × kv_heads × head × n_layers × dtype。
        free, _ = torch.cuda.mem_get_info()
        per_block_bytes = (2 * 2 * self.DEFAULT_BLOCK_SIZE * self.kv_num_heads
                           * self.head_size * self.num_layers * torch.finfo(self.dtype).bits // 8)
        # 余量：权重(模型 bf16 ~1.2GB) + graph/logits buffer(max_batch×vocab) + activation。
        # max_batch=512 时 _logits buffer ≈ 155MB，graph workspace 等，保守留 6GB。
        kv_budget = max(free - 6 * (1 << 30), per_block_bytes * 16)
        n_blocks = max(int(kv_budget // per_block_bytes),
                       max_batch_size * 2)  # 至少够 max_batch 条短请求各 2 block
        logger.info(f"KV 预算: free={free/1e9:.1f}GB 留6GB → n_blocks={n_blocks} "
                    f"({n_blocks*self.DEFAULT_BLOCK_SIZE} tokens, 可跑 {n_blocks//max_seq_blocks} 条满{self.max_position}上下文)")
        self.cache_manager = KVCacheManager(
            n_blocks=n_blocks, block_size=self.DEFAULT_BLOCK_SIZE,
            n_layers=self.num_layers, n_heads=self.kv_num_heads, head_size=self.head_size,
            dtype=self.dtype, device=self.device, max_batch_size=max_batch_size,
            max_tokens=self.max_position
        )
        self.graph_runner = ModelGraphRunner(
            model=self.model, num_layers=self.num_layers, num_heads=self.num_heads,
            head_size=self.head_size, kv_num_heads=self.kv_num_heads,
            hidden_dim=self.config.hidden_size, intermediate_size=self.intermediate_size,
            device=self.device, max_batch_size=max_batch_size, dtype=self.dtype,
            max_context_length=self.max_position
        )

        self.prefill_runner = ModelPrefillRunner(
            model=self.model, num_layers=self.num_layers, num_heads=self.num_heads,
            head_size=self.head_size, kv_num_heads=self.kv_num_heads,
            hidden_dim=self.config.hidden_size, intermediate_size=self.intermediate_size,
            device=self.device, max_batch_size=max_batch_size, dtype=self.dtype,
            max_context_length=self.max_position
        )
        self.scheduler = Scheduler(max_batch_size, max_prefill_tokens, self.tokenizer)
        # chunked prefill 仅对支持的架构启用（Qwen3 GQA+with_kvcache），默认 chunk=512。
        # 不支持的架构（DeepSeek MLA prefill 不读 cache 前缀）保持 max_chunk_tokens=1024
        #（= max_position，prompt ≤1024 整条 prefill，不触发 chunked，行为正确）。
        if self.adapter.supports_chunked_prefill(self.config):
            self.scheduler.max_chunk_tokens = min(512, self.max_position)
        else:
            self.scheduler.max_chunk_tokens = self.max_position
        self.sampler = Sampler()
        self._decode_ctx = DecodeContext()
        self._stream_event: Optional[StreamEvent] = None
        # decode batch 脏标志：True 时 prepare() 重建元数据。稳定 decode 每步 batch
        # 成员/顺序不变，仅当有序列完成 / prefill 新进 / append 跨 block 分配时置脏，
        # 避免每步构建 512 元素列表 + 比较的 ~1.2ms CPU 开销（见 DecodeContext.prepare）。
        self._ctx_batch_dirty = True
        
        # 状态
        # eos 兜底：Qwen 旧版 tokenizer 的 eos_token_id 可能为 None，
        # 此时从 model.generation_config 读取（如 Qwen-Chat 的 151643）
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = getattr(self.model.generation_config, 'eos_token_id', None)
        self.stream_callbacks = {}
        # 非流式 /generate 等待结果：seq_id -> (asyncio.Future, 完整文本)
        # 后台 rank0_inference_loop 在 update_sequences 里 set_result，HTTP handler await。
        # 这样多个并发 /generate 共享同一个 scheduler batch = continuous batching。
        self._completion_futures: Dict[int, "asyncio.Future"] = {}
        self._completion_results: Dict[int, str] = {}

        # 捕获 CUDA Graph：两架构均 graph-friendly，一个 graph 通吃所有 ≤1024 序列。
        logger.info("Capturing CUDA Graphs...")
        # 细粒度桶（对齐 nano-vllm）：1,2,4,8 + 16 步等差到 max_batch_size。
        # padding 率恒 ≤15/bs（bs=256 时最多 pad 到 272，约 6%），变长 batch 收尾无大浪费。
        # 旧 1.5x 粗桶（48,72,108,162,243,365,512）在 bs=256 时 pad 到 365 浪费 42%、
        # bs=388 时 pad 到 512 浪费 32%，是 1000 请求吞吐落后 nano 的主因之一。
        cap_sizes = [b for b in [1, 2, 4, 8] if b <= max_batch_size]
        cap_sizes += list(range(16, max_batch_size + 1, 16))
        cap_sizes = sorted(set(cap_sizes))
        # 同步 scheduler 的 pad 档位：decode batch 向上取整 padding 必须落到已捕获的
        # graph batch_size，否则 graph key 不命中。单一来源 = engine 的 cap_sizes。
        self.scheduler.batch_sizes = cap_sizes
        self.graph_runner.capture(self.cache_manager, batch_sizes=cap_sizes)
        logger.info("CUDA Graphs captured.")
        # 预热 sampler 编译路径（torch.compile reduce-overhead 首次调用每个 shape
        # 会捕获 CUDA Graph ~1-2s）。不预热时首个多 batch prefill/decode 会卡秒级。
        # 对所有 cap_sizes 用 temp>0 路径（触发 _compiled_sample）各跑一次。
        self._warmup_sampler(cap_sizes)
        # 预热 prefill eager 路径：cuBLAS/flash 首次跑每个 (B,S) shape 会选算法/编译，
        # 首个真实多 batch prefill 多耗 ~100-200ms。用短 dummy prompt 在主 batch size 跑一次。
        # 仅对支持 chunked prefill 的架构（Qwen3 GQA）——DeepSeek MLA prefill 用
        # flash_attn_func 自包含路径，dummy prefill 触发 cudaErrorAssert，跳过。
        if self.adapter.supports_chunked_prefill(self.config):
            self._warmup_prefill(cap_sizes)

        # 注册退出钩子
        atexit.register(self.shutdown)

    def _warmup_sampler(self, batch_sizes):
        """对所有捕获的 batch_size 预热 sampler 编译路径，消除首次调用的 ~1-2s 捕获开销。

        greedy（argmax）路径无编译开销；此处预热的是 temp>0 的 _compiled_sample 路径，
        覆盖连续批处理下任意 batch_size 首次采样。"""
        vocab = self.config.vocab_size
        dtype = next(self.model.parameters()).dtype
        for bs in batch_sizes:
            fake_logits = torch.zeros(bs, vocab, dtype=dtype, device=self.device)
            temps = torch.full((bs,), 0.01, device=self.device)
            topp = torch.ones(bs, device=self.device)
            rep = torch.ones(bs, device=self.device)
            self.sampler(fake_logits, temps, topp, 1000,
                         prev_tokens=None, rep_penalties=rep,
                         all_greedy=False, any_rep_pen=False)
        # 也预热 repetition-penalty 路径（bs=1 足矣，scatter 形状随 vocab 而靘认 prev_tokens 长度）
        fake_logits = torch.zeros(1, vocab, dtype=dtype, device=self.device)
        prev = torch.tensor([[0, 1, 2]], dtype=torch.long, device=self.device)
        rep = torch.tensor([1.1], device=self.device)
        self.sampler(fake_logits, torch.tensor([0.01], device=self.device),
                     torch.ones(1, device=self.device), 1000,
                     prev_tokens=prev, rep_penalties=rep,
                     all_greedy=False, any_rep_pen=True)

    def _warmup_prefill(self, batch_sizes):
        """预热 prefill eager 路径，消除 cuBLAS/flash 首次跑每个 (B,S) shape 的算法选择开销。

        用极短 dummy prompt（~8 token）在代表性 batch size 跑一次 prefill+少量 decode，
        然后立即释放这些 dummy seq 的 KV block。代表 batch 取 cap_sizes 中 ≤32 的若干档
        （大 batch 算法选择与小 batch 同路径，无需每档都跑）。"""
        # 选代表性 batch size：小 bs 密集测，大 bs 取一档（64）即可覆盖 cuBLAS 大矩阵路径。
        warm_sizes = [b for b in batch_sizes if b <= 32]
        if 64 in batch_sizes:
            warm_sizes.append(64)
        dummy_prompt = "warmup "  # ~2 token
        sid_base = 10_000_000  # 避免与真实 seq_id 冲突
        for bs in warm_sizes:
            for i in range(bs):
                seq = Sequence(sid_base + i, dummy_prompt, self.tokenizer, max_tokens=2)
                seq.temperature = 0.01; seq.top_p = 1.0
                if self.eos_token_id is not None:
                    seq.eos_token_id = self.eos_token_id
                self.scheduler.add_request(seq)
            # 跑 prefill + 1 decode（触发该 batch_size 的 prefill GEMM/flash + decode graph replay）
            for _ in range(20):
                b, bt = self.get_next_batch()
                if not b:
                    break
                ctx = BatchInferenceContext(len(b), bt, b)
                self.step(ctx); self.collect(ctx); self.update_sequences(ctx.sequences)
                if not self.scheduler.running_sequences and not self.scheduler.waiting_queue:
                    break
            # 清理 dummy seq 的 KV（释放 block）
            for i in range(bs):
                try:
                    self.cache_manager.free(sid_base + i)
                except Exception:
                    pass
            self.scheduler.running_sequences.clear()
            self.scheduler.finished_sequences.clear()

    def _init_distributed(self):
        setup()
        self.rank = get_rank()
        torch.cuda.set_device(self.rank)
        self.device_str = f"cuda:{self.rank}"

    def _init_model(self, model_path: str):
        logger.info(f"Loading model {model_path} on rank {self.rank}")
        self.model, self.tokenizer = load_model(model_path, device=self.device_str)
        self.model.eval()
        self.config = self.model.config

    def _init_config(self):
        # 通过适配器提取架构相关维度（GQA vs MLA 等差异在此屏蔽）
        from models import build_adapter
        self.adapter = build_adapter(self.config)

        self.num_layers = self.adapter.num_layers(self.config)
        g_num_heads, g_kv_heads, cache_head_size = self.adapter.cache_dims(self.config)

        # 张量并行切分（按 head 切）
        world_size = get_world_size()
        assert g_num_heads % world_size == 0, f"num_heads {g_num_heads} 不可被 world_size {world_size} 整除"
        # MLA: kv 是单 latent head（不可按 head 切），保持 1
        if g_kv_heads == 1:
            assert world_size == 1 or self.adapter.model_type != "deepseek", \
                "DeepSeek MLA 单 latent head，TP>1 需对 latent 切分（首版仅支持 TP=1）"
        else:
            assert g_kv_heads % world_size == 0

        self.num_heads = g_num_heads // world_size
        self.kv_num_heads = g_kv_heads if g_kv_heads == 1 else g_kv_heads // world_size
        # head_size = KV cache 存储维度（GQA=head_size, MLA=latent_dim）
        self.head_size = cache_head_size
        self.intermediate_size = self.adapter.intermediate_size(self.config, world_size)

    def _auto_configure(self) -> Tuple[str, torch.dtype]:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype == torch.bfloat16:
            self.model.to(torch.bfloat16)
        return "cuda", dtype

    def shutdown(self):
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
                    temperature: float = 0.7, top_p: float = 0.9,
                    repetition_penalty: float = 1.0, stop=None) -> int:
        # 上下文上限：钳 prompt+gen ≤ max_position（可配，DeepSeek 钳到 1024）。
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        cap = self.max_position
        if len(prompt_ids) > cap:
            prompt_ids = prompt_ids[:cap]
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            logger.warning(f"上下文上限 {cap}：prompt 过长({len(prompt_ids)}>{cap})，已截断")
        max_tokens = max(1, min(max_tokens, cap - len(prompt_ids)))
        seq_id = hash(prompt + str(time.time())) % (2 ** 32)
        seq = Sequence(seq_id, prompt, self.tokenizer, max_tokens)
        seq.temperature = temperature
        seq.top_p = top_p
        seq.repetition_penalty = repetition_penalty
        seq.stop_strings = list(stop) if stop else []
        # 同步 engine 解析出的 eos_token_id（覆盖 Sequence 中可能为 None 的默认值）
        if self.eos_token_id is not None:
            seq.eos_token_id = self.eos_token_id
        self.scheduler.add_request(seq)
        return seq_id

    def register_stream_callback(self, seq_id: int, callback):
        self.stream_callbacks[seq_id] = callback

    def unregister_stream_callback(self, seq_id: int):
        self.stream_callbacks.pop(seq_id, None)

    def new_completion_future(self, seq_id: int) -> "asyncio.Future":
        """为非流式 /generate 创建完成 Future。后台循环在 update_sequences 里
        set_result(完整文本)。HTTP handler await 这个 Future 即可拿到结果。
        多个并发 /generate 共享同一个 scheduler batch = continuous batching。"""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._completion_futures[seq_id] = fut
        return fut

    def _complete_seq(self, seq: Sequence):
        """seq 完成时调用：decode 完整 output 存结果并唤醒等待的 HTTP handler。"""
        try:
            text = self.tokenizer.decode(seq.output_ids, skip_special_tokens=True)
        except Exception:
            text = ""
        self._completion_results[seq.seq_id] = text
        fut = self._completion_futures.pop(seq.seq_id, None)
        if fut is not None and not fut.done():
            fut.set_result(text)

    def get_next_batch(self) -> Tuple[List[Sequence], str]:
        return self.scheduler.get_next_batch()

    @torch.inference_mode()
    def step(self, ctx: BatchInferenceContext) -> bool:
        if not ctx.sequences: return False
        if ctx.batch_type == "prefill":
            self._prefill(ctx.sequences)
            # prefill 产出新 seq 进入 decode → 下一步 decode batch 成员变化，置脏
            self._ctx_batch_dirty = True
        else:
            self._decode(ctx)
        return True

    @torch.inference_mode()
    def launch(self, ctx: BatchInferenceContext):
        """Decode 专用 Phase 1：append + prepare + 提交 forward，GPU 开始异步执行后立刻返回。"""
        batch = ctx.sequences
        batch_size = len(batch)
        seen = set()
        for seq in batch:
            if seq.seq_id not in seen:
                self.cache_manager.append(seq.seq_id)
                seen.add(seq.seq_id)
        # append 跨 block 边界分配新块时标记 _dirty_seqs → 下一步 prepare 需重建 block_table
        if self.cache_manager._dirty_seqs:
            self._ctx_batch_dirty = True
        input_ids = self._decode_ctx.prepare(batch, self.device, self.cache_manager,
                                             batch_dirty=self._ctx_batch_dirty)
        self._ctx_batch_dirty = False  # 本步已处理；后续 append/finished 会按需重新置脏
        ctx.logits = self.graph_runner.forward(input_ids, self.cache_manager, batch_size)

    @torch.inference_mode()
    def collect(self, ctx: BatchInferenceContext):
        if ctx.batch_type == "prefill":
            return
        """Decode 专用 Phase 2：flush + seqlens+1 + sample + commit（GPU forward 已完成）。"""
        batch, bs, logits = ctx.sequences, ctx.batch_size, ctx.logits
        if rank0():
            self._flush_stream()
        if rank0():
            dctx = self._decode_ctx
            next_tokens_gpu = self.sampler(
                logits, dctx.temps, dctx.topp, 50,
                prev_tokens=dctx.prev_tokens, rep_penalties=dctx.rep_penalties,
                all_greedy=dctx.all_greedy, any_rep_pen=dctx.any_rep_pen)
            dctx.commit(next_tokens_gpu, self.graph_runner._input_ids, bs, batch)
            # 把本步新 token 追加进 prev_tokens，供下一步 repetition penalty 使用
            # 用 CPU 侧 any_rep_pen 标志避免 torch.any GPU→CPU 同步
            if dctx.prev_tokens is not None and dctx.any_rep_pen:
                new_col = next_tokens_gpu.unsqueeze(1)              # [bs, 1]
                dctx.prev_tokens = torch.cat(
                    [dctx.prev_tokens, new_col], dim=1)             # [bs, L+1]

    def _decode(self, ctx: BatchInferenceContext):
        """同步 decode：launch + collect，供 step() 和 non-rank0 路径调用。"""
        self.launch(ctx)
        batch, bs, logits = ctx.sequences, ctx.batch_size, ctx.logits
        self.cache_manager.commit(bs)

    def generate(self, prompts: List[str], max_tokens: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9,
                 repetition_penalty: float = 1.0, stop=None) -> Dict[str, str]:
        seq_ids = [self.add_request(p, max_tokens, temperature=temperature, top_p=top_p,
                                    repetition_penalty=repetition_penalty, stop=stop) for p in prompts]
        seq_map = {sid: p for sid, p in zip(seq_ids, prompts)}
        # 清理上轮残留的已完成序列，避免 get_finished_results 返回过期 seq 导致 KeyError
        self.scheduler.finished_sequences.clear()

        # 简易事件循环（对齐 api_server 的 rank0 推理循环语义）
        for _ in range(max_tokens * len(prompts) + 64):  # 安全上限
            batch, batch_type = self.get_next_batch()

            # waiting：请求未攒够批次，等 prefill_timeout 到期再调度
            if batch_type == "waiting" or not batch:
                if not self.scheduler.running_sequences and not self.scheduler.waiting_queue:
                    break
                time.sleep(0.001)
                continue

            ctx = BatchInferenceContext(len(batch), batch_type, batch)
            self.step(ctx)
            self.collect(ctx)
            self.update_sequences(ctx.sequences)

            if not self.scheduler.running_sequences and not self.scheduler.waiting_queue:
                break

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

        # 1. 准备：变长拼接。每条 seq 取本 step 的 chunk，拼成 1D input_ids [total_tokens]。
        stats.prep_time = time.time()
        device = self.device
        from models.base import PrefillMeta

        # 每条 seq 的本 chunk token + 起始 offset（已 prefill 的 token 数）
        chunk_tokens = [seq.get_next_input_ids() for seq in batch]
        chunk_lens = [len(t) for t in chunk_tokens]
        offsets = [s.prefill_done for s in batch]
        need_sample = [s._chunk_is_last for s in batch]
        n_seqs = len(batch)

        # 2. KV Cache 分配：仅第一次 prefill（prefill_done==0）按整个 prompt 长度预分配 block；
        #    续切 chunk 复用已分配 block。
        for seq in batch:
            if seq.prefill_done == 0:
                ok, _ = self.cache_manager.alloc(seq.seq_id, len(seq.input_ids))
                if not ok:
                    raise RuntimeError("OOM: prefill alloc failed")

        # 3. 组装变长元数据（cu_seqlens / position_ids / slot_mapping / block_table）。
        #    cu_seqlens/offsets 在 CPU 算好后一次 H2D；slot_mapping 经 block_table 矩阵索引
        #    在 GPU 向量化算（无逐 token Python 循环，512 seqs 时省 ~20ms）。
        total_tokens = sum(chunk_lens)
        cm = self.cache_manager
        block_size = cm.block_size

        # cu_seqlens_q/k + offsets（CPU 列表 → tensor）
        cu_q = [0]
        cu_k = [0]
        for off, clen in zip(offsets, chunk_lens):
            cu_q.append(cu_q[-1] + clen)
            cu_k.append(cu_k[-1] + off + clen)
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)
        max_seqlen_q = max(chunk_lens)
        max_seqlen_k = max((o + c for o, c in zip(offsets, chunk_lens)), default=0)

        # block_table：复用 cache_manager 的 triton kernel 填充（context_lens=kv_len 仅用于建表）
        block_table = cm._block_table_buffer[:n_seqs]
        cm.cache_batch_data([s.seq_id for s in batch], [cu_k[i + 1] - cu_k[i] for i in range(n_seqs)])

        # input_ids + position_ids：numpy 拼接（比 Python list extend + torch.tensor 快）后一次 H2D
        import numpy as np
        ids_np = np.empty(total_tokens, dtype=np.int64)
        pos_np = np.empty(total_tokens, dtype=np.int64)
        p = 0
        for toks, off, clen in zip(chunk_tokens, offsets, chunk_lens):
            ids_np[p:p + clen] = toks
            pos_np[p:p + clen] = np.arange(off, off + clen)
            p += clen
        input_ids = torch.from_numpy(ids_np).to(device)
        position_ids = torch.from_numpy(pos_np).to(device)

        # slot_mapping：向量化。每 token 的 seq 归属 = searchsorted(cu_seqlens_q)；
        # abs_pos = offset[seq] + (token - cu_q[seq])；slot = bt[seq, abs//bs]*bs + abs%bs。
        offsets_gpu = torch.tensor(offsets, dtype=torch.int32, device=device)
        token_idx = torch.arange(total_tokens, device=device)
        seq_of_token = torch.searchsorted(cu_seqlens_q[1:], token_idx, right=True)
        local_pos = token_idx - cu_seqlens_q[seq_of_token]
        abs_pos = offsets_gpu[seq_of_token].long() + local_pos
        block_idx = abs_pos // block_size
        offset_in_block = abs_pos % block_size
        slot_mapping = (block_table[seq_of_token, block_idx] * block_size
                        + offset_in_block).to(torch.int32)

        meta = PrefillMeta(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            position_ids=position_ids, slot_mapping=slot_mapping,
            block_table=block_table, n_seqs=n_seqs,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        )
        stats.prep_time = time.time() - stats.prep_time

        # 4. 推理（变长：input_ids 1D，logits 末 token 用于采样）
        stats.gpu_time = time.time()
        logits = self.prefill_runner.forward(input_ids, self.cache_manager, meta)
        stats.gpu_time = time.time() - stats.gpu_time

        # 5. 采样：仅对最后 chunk（need_sample）采样首 token；中间 chunk 不采样、不转 decode。
        #    logits 是 [total_tokens, vocab]，每条 seq 取其本 chunk 末 token（cu_q 边界）。
        if rank0():
            stats.sample_time = time.time()
            sample_idx = [i for i, ns in enumerate(need_sample) if ns]
            if sample_idx:
                sample_batch = [batch[i] for i in sample_idx]
                # 每条 sample seq 在 1D logits 中的末 token 位置 = cu_q[i+1] - 1
                last_pos = [int(cu_q[i + 1]) - 1 for i in sample_idx]
                last_logits = logits[last_pos, :]
                temps_list = [s.temperature for s in sample_batch]
                rep_list = [getattr(s, 'repetition_penalty', 1.0) for s in sample_batch]
                temps = torch.tensor(temps_list, device=device)
                topp = torch.tensor([s.top_p for s in sample_batch], device=device)
                rep_pen = torch.tensor(rep_list, device=device)
                any_rep = any(r > 1.0 for r in rep_list)
                all_greedy = all(t <= 0 for t in temps_list)
                prev = None
                if any_rep:
                    hist = [list(s.input_ids) for s in sample_batch]
                    max_l = max(len(h) for h in hist)
                    prev = torch.tensor(
                        [h + [-1] * (max_l - len(h)) for h in hist], dtype=torch.long, device=device)
                next_tokens = self.sampler(last_logits, temps, topp, 1000,
                                           prev_tokens=prev, rep_penalties=rep_pen,
                                           all_greedy=all_greedy, any_rep_pen=any_rep).tolist()
                for j, seq in enumerate(sample_batch):
                    seq.advance_prefill(seq._chunk_len)
                    seq._next_token = next_tokens[j]
                    seq._chunk_sampled = True
            # 中间 chunk：清空 _next_token，update_sequences 据此走 advance_prefill（不转 decode）
            for i, ns in enumerate(need_sample):
                if not ns:
                    batch[i]._next_token = None
                    batch[i]._chunk_sampled = False
            stats.sample_time = time.time() - stats.sample_time

            stats.total_time = time.time() - stats.total_time
            logger.info(f"Prefill: Prep {stats.prep_time*1000:.1f}ms, GPU {stats.gpu_time*1000:.1f}ms, "
                        f"Total {stats.total_time*1000:.1f}ms | n_seqs={n_seqs} total_tokens={total_tokens} "
                        f"chunks={chunk_lens} offsets={offsets} sample={sum(need_sample)}")


    def update_sequences(self, sequences: List[Sequence]):
        # decode 稳态快速路径：无流式 client、无 stop 串、全 decode seq 时，
        # 每步 512 seq 循环里 rank0()/dict.get()/方法调用等 ~10 个 Python 操作全是浪费
        # （实测 0.75ms/步）。快速路径只做 append+position+finished 判断；
        # 发现任何慢路径条件（流式/stop串/prefill seq）立即回退完整路径重跑。
        if not self.stream_callbacks and rank0():
            # 先扫条件再应用：中途回退完整路径时不能留下已修改的 seq（否则 double append）
            fast = True
            for seq in sequences:
                if seq._next_token is None or seq.state != "decode" or seq.stop_strings:
                    fast = False
                    break
            if fast:
                finished_any = False
                seen = set()  # decode batch 含循环复制的 pad 重复 seq，须去重（对齐完整路径 seq_dict）
                for seq in sequences:
                    if seq.seq_id in seen:
                        continue
                    seen.add(seq.seq_id)
                    next_token = seq._next_token
                    seq.output_ids.append(next_token)
                    seq.full_ids.append(next_token)
                    seq.current_position += 1
                    if len(seq.output_ids) >= seq.max_tokens or next_token == seq.eos_token_id:
                        seq._finished = True
                        seq.state = "finished"
                        finished_any = True
                if finished_any:
                    freed = set()  # padded batch 含重复 seq，须去重（否则 mark_finished 多次→finished 重复）
                    for seq in sequences:
                        if seq.state == "finished" and seq.seq_id not in freed:
                            freed.add(seq.seq_id)
                            self.cache_manager.free(seq.seq_id)
                            self.scheduler.mark_finished(seq)
                            self._complete_seq(seq)
                    self._ctx_batch_dirty = True
                    self._flush_stream()
                return
        seq_dict = defaultdict(int)
        output_tokens = []
        callbacks_batch = []
        stream_callbacks = self.stream_callbacks
        tokenizer = self.tokenizer

        finished_any = False
        for seq in sequences:
            next_token = seq._next_token
            if next_token is None:
                # chunked prefill 中间 chunk：不产生 token，仅推进 prefill_done（KV 已写入 cache）
                if seq.state == "prefill" and getattr(seq, "_chunk_sampled", False) is False and seq._chunk_len > 0:
                    seq.advance_prefill(seq._chunk_len)
                    seq._chunk_len = 0
                continue

            seq_id = seq.seq_id
            if seq_id in seq_dict:
                continue

            seq_dict[seq_id] = seq.current_position
            seq.update_state(next_token, None)

            # 服务端停止字符串：命中即把 output 截断到停止边界并标记完成。
            # 这避免 client 在停止边界提前断流后、server 仍继续生成导致的 seq 孤儿
            # （下一个请求的 prefill/decode 会与孤儿 seq 共用常驻 block_table/seqlens 缓冲 → 状态错乱）。
            if seq.stop_strings and not seq.is_finished():
                text = tokenizer.decode(seq.output_ids, skip_special_tokens=True)
                hit_idx = -1
                hit_len = 0
                for s in seq.stop_strings:
                    i = text.find(s)
                    if i != -1 and (hit_idx == -1 or i < hit_idx):
                        hit_idx = i
                        hit_len = len(s)
                if hit_idx != -1:
                    # 截断 output_ids：近似按字符数砍（base 模型停止串通常对齐 token 边界，
                    # 多砍一两个 token 不影响展示，client 还会再做一次精确截断）。
                    keep_chars = hit_idx
                    # 反推保留的 token 数：逐 token decode 直到覆盖 keep_chars
                    kept = 0
                    for k in range(1, len(seq.output_ids) + 1):
                        if len(tokenizer.decode(seq.output_ids[:k], skip_special_tokens=True)) >= keep_chars:
                            kept = k
                            break
                    if kept > 0:
                        seq.output_ids = seq.output_ids[:kept]
                        seq.full_ids = seq.input_ids + seq.output_ids
                        seq.current_position = len(seq.input_ids) + len(seq.output_ids)
                    seq.state = "finished"
                    seq._stop_hit = True
                    seq._finished = True

            if rank0():
                cb = stream_callbacks.get(seq_id)
                if cb:
                    output_tokens.append(next_token)
                    callbacks_batch.append(cb)

            finished_this_step = seq.is_finished() or getattr(seq, "_stop_hit", False)
            if finished_this_step:
                finished_any = True
                self.cache_manager.free(seq_id)
                if rank0():
                    self.scheduler.mark_finished(seq)
                    # 唤醒等待该 seq 的非流式 /generate handler（continuous batching）
                    self._complete_seq(seq)

        if rank0():
            self._stream_event = StreamEvent(output_tokens, callbacks_batch) if callbacks_batch else None
            # 若本步有 seq 结束（EOS / stop 串 / max_tokens），seq 会被移出 running，
            # 下一轮 get_next_batch 不再包含它 → collect() 的 _flush_stream 不会执行，
            # 本步暂存的最后一批 token 会丢失。这里立即冲刷，保证流式 client 收到完整输出。
            if finished_any:
                self._ctx_batch_dirty = True  # batch 成员变化 → 下一步 prepare 重建
                self._flush_stream()

    def _flush_stream(self):
        """
        在 GPU 执行 graph 的空窗口内，将上一步暂存的 StreamEvent 解码并推送给流式客户端。
        由 _decode 在 replay() 之后、sampler() 之前调用，单线程无 GIL 竞争。
        """
        evt = self._stream_event
        if evt is None:
            return
        self._stream_event = None
        texts = self.tokenizer.batch_decode(
            [[t] for t in evt.tokens], skip_special_tokens=True
        )
        for cb, token, txt in zip(evt.callbacks, evt.tokens, texts):
            try:
                cb(token, txt)
            except Exception:
                pass