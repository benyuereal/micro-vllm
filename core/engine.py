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

    def __init__(self, model_path: str, max_batch_size: int = 256, max_prefill_tokens: int = 2048):
        self._init_distributed()
        self._init_model(model_path)
        self._init_config()

        # 核心组件初始化
        self.device, self.dtype = self._auto_configure()

        # 序列长度上限：两架构统一固定 1024。max_tokens 决定 block_table 列数
        # (max_seq_blocks=4)，与 DeepSeek attention 的 arange(1024) 精确对齐。总长
        # prompt+gen 由 add_request 钳到 ≤1024。>1024 长序列留待 tile op 恢复。
        self.max_position = 1024
        # n_blocks 动态算：max_batch_size 序列各占满 max_position + prefill 余量。
        # 每序列最多 ceil(1024/256)=4 block。余量给 prefill 临时分配（同长度组一次性 prefill）。
        max_seq_blocks = (self.max_position + self.DEFAULT_BLOCK_SIZE - 1) // self.DEFAULT_BLOCK_SIZE
        n_blocks = max_batch_size * max_seq_blocks + max_batch_size  # +1 block/序列余量
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
            device=self.device, max_batch_size=max_batch_size, dtype=self.dtype
        )

        self.prefill_runner = ModelPrefillRunner(
            model=self.model, num_layers=self.num_layers, num_heads=self.num_heads,
            head_size=self.head_size, kv_num_heads=self.kv_num_heads,
            hidden_dim=self.config.hidden_size, intermediate_size=self.intermediate_size,
            device=self.device, max_batch_size=max_batch_size, dtype=self.dtype
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
        if self.device == "cuda":
            logger.info("Capturing CUDA Graphs...")
            # 离散捕获到 max_batch_size：小 bs 密集（1,2,4,8,16,32），大 bs 按 2 倍（64,128,256）。
            # replay 时向下取整到 >= 实际 bs 的最小捕获值，padding 复用已有 seq。
            cap_sizes = [b for b in [1, 2, 4, 8, 16, 32] if b <= max_batch_size]
            b = 64
            while b <= max_batch_size:
                cap_sizes.append(b)
                b *= 2
            if not cap_sizes or cap_sizes[-1] < max_batch_size:
                cap_sizes.append(max_batch_size)
            self.graph_runner.capture(self.cache_manager, batch_sizes=cap_sizes)
            logger.info("CUDA Graphs captured.")
        else:
            logger.info("非 CUDA 设备，跳过 CUDA Graph 捕获（eager 路径）。")
            
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
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if dtype == torch.bfloat16:
                self.model.to(torch.bfloat16)
            return "cuda", dtype
        return "cpu", torch.float32

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
        # 固定 1024 上下文：钳 prompt+gen ≤ max_position，否则 decode 越界。
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        cap = self.max_position
        if len(prompt_ids) > cap:
            prompt_ids = prompt_ids[:cap]
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            logger.warning(f"固定 {cap} 上下文：prompt 过长({len(prompt_ids)}>{cap})，已截断")
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
        input_ids = self._decode_ctx.prepare(batch, self.device, self.cache_manager)
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
                prev_tokens=dctx.prev_tokens, rep_penalties=dctx.rep_penalties)
            dctx.commit(next_tokens_gpu, self.graph_runner._input_ids, bs, batch)
            # 把本步新 token 追加进 prev_tokens，供下一步 repetition penalty 使用
            if dctx.prev_tokens is not None and torch.any(dctx.rep_penalties > 1.0):
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

        # 1. 准备
        stats.prep_time = time.time()
        device = self.device

        # chunked prefill：每条 seq 取本 step 的 chunk（get_next_input_ids 按 prefill_done+_chunk_len 切片）
        input_ids = torch.tensor([seq.get_next_input_ids() for seq in batch], dtype=torch.long, device=device)
        seq_lens = [len(ids) for ids in input_ids]
        # position_offsets：每条 seq 已 prefill 的 token 数（KV 续写起始位置 / RoPE 位置偏移）
        position_offsets = torch.tensor([s.prefill_done for s in batch], dtype=torch.int32, device=device)
        # 本 step 是否产生 token：仅最后 chunk（或完整短 prompt）采样
        need_sample = [s._chunk_is_last for s in batch]

        # 2. KV Cache 分配：仅第一次 prefill（prefill_done==0）按整个 prompt 长度预分配 block；
        #    续切 chunk 复用已分配 block，不重新 alloc。
        for seq in batch:
            if seq.prefill_done == 0:
                ok, _ = self.cache_manager.alloc(seq.seq_id, len(seq.input_ids))
                if not ok:
                    raise RuntimeError("OOM: prefill alloc failed")

        # 重建 block_table + 设 cache_seqlens=position_offsets（chunked 续写位置）
        self.cache_manager.cache_batch_data(
            [s.seq_id for s in batch], [int(o) for o in position_offsets.tolist()])
        stats.prep_time = time.time() - stats.prep_time

        # 3. 推理
        stats.gpu_time = time.time()
        logits = self.prefill_runner.forward(input_ids, self.cache_manager, len(batch),
                                             position_offsets=position_offsets)
        stats.gpu_time = time.time() - stats.gpu_time

        # 4. 采样：仅对最后 chunk（need_sample）采样首 token；中间 chunk 不采样、不转 decode
        if rank0():
            stats.sample_time = time.time()
            sample_idx = [i for i, ns in enumerate(need_sample) if ns]
            if sample_idx:
                sample_batch = [batch[i] for i in sample_idx]
                temps = torch.tensor([s.temperature for s in sample_batch], device=device)
                topp = torch.tensor([s.top_p for s in sample_batch], device=device)
                rep_pen = torch.tensor(
                    [getattr(s, 'repetition_penalty', 1.0) for s in sample_batch], device=device)
                prev = None
                if torch.any(rep_pen > 1.0):
                    hist = [list(s.input_ids) for s in sample_batch]
                    max_l = max(len(h) for h in hist)
                    prev = torch.tensor(
                        [h + [-1] * (max_l - len(h)) for h in hist], dtype=torch.long, device=device)
                next_tokens = self.sampler(logits[sample_idx, -1, :], temps, topp, 1000,
                                           prev_tokens=prev, rep_penalties=rep_pen).tolist()
                for j, seq in enumerate(sample_batch):
                    # 最后 chunk：先把 prefill_done/current_position 推进到整个 prompt 末尾，
                    # 随后 update_sequences 的 update_state(+1) 才能给出正确的 decode 位置。
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
                        f"Total {stats.total_time*1000:.1f}ms | Batch {len(batch)} chunks={seq_lens} "
                        f"offsets={position_offsets.tolist()} sample={sum(need_sample)}")


    def update_sequences(self, sequences: List[Sequence]):
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