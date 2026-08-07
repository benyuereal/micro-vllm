import logging
import torch
import torch.nn.functional as F
from typing import Dict, List

from core.paged_attention import PagedAttention
from kernel.matmul import matmul_v3
from kernel.rmsnorm import rmsnorm_, rmsnorm_residual_gemm as rmsnorm_residual
from kernel.swiglu import matmul_swiglu
from .rope import RoPE
from core.parallel_config import get_rank, get_world_size, all_reduce
from models import build_adapter
import torch._dynamo

torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.cache_size_limit = 128

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelGraphRunner:
    def __init__(self, model, num_layers: int, num_heads: int, head_size: int,
                 kv_num_heads: int, hidden_dim: int, intermediate_size: int,
                 device: str, max_batch_size: int = 32, dtype: torch.dtype = torch.bfloat16,
                 top_k: int = 1000):
        self.model = model
        self.num_layers = num_layers
        self.rank, self.world_size = get_rank(), get_world_size()

        # 模型维度
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.intermediate_size = intermediate_size
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.vocab_size = model.config.vocab_size
        self.device = device
        self.max_bs = max_batch_size
        self.dtype = dtype
        self.top_k = top_k

        # 架构适配器
        self.adapter = build_adapter(model.config)
        # 架构相关标量（DeepSeek MLA 的 softmax_scale 等）
        self._ds_softmax_scale = self.adapter.softmax_scale(model.config)

        # 通用模块
        # PagedAttention 的 head 维度 = KV cache 存储维度（GQA=head_size, MLA=latent_dim）
        # rope_dim = RoPE 实际作用维度（GQA=head_size, MLA=qk_rope_head_dim）
        self.attention = PagedAttention(num_heads, head_size, kv_num_heads, device, max_batch_size,
                                        rope_dim=self.adapter.rope_dim(model.config))
        self.rope = RoPE()

        # 编译函数
        self._fast_mlp = self._compile_fn(self._mlp)

        # 初始化
        self.adapter.prepare_weights(self.model, self.world_size, self.rank)
        self._alloc_bufs()

        # CUDA Graph
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._is_graph_ready = False

    def _compile_fn(self, fn):
        return torch.compile(
            fn,
            fullgraph=True,
            backend="inductor",
            options={
                "max_autotune": True,
                "max_autotune_gemm": True,
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            }
        )

    @staticmethod
    def _mlp(x, gu_weight, d_weight):
        gate_up = x @ gu_weight
        up, gate = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return activated @ d_weight

    def _alloc_bufs(self):
        max_b = self.max_bs
        self._input_ids = torch.empty(max_b, dtype=torch.long, device=self.device)
        self._logits = torch.empty(max_b, self.vocab_size, dtype=self.dtype, device=self.device)

        # 由 adapter 决定 buffer 形状（不同架构需要不同的中间张量）
        bufs = self.adapter.alloc_bufs(self.model, max_b, self.hidden_dim, self.dtype, self.device)
        self._h_buf = bufs["_h_buf"]
        self._qkv = bufs["_qkv"]
        self._attn_out = bufs["_attn_out"]
        self._residual = bufs["_residual"]
        self._swiglu_out = bufs.get("_swiglu_out")  # Qwen 用；DeepSeek 可能不用

    # ==========================================
    # 主推理逻辑
    # ==========================================

    def decode(self, input_ids, bs, cache_manager, block_table):
        embed = self.adapter.embed(self.model)
        blocks = self.adapter.blocks(self.model)
        h = embed(input_ids)
        fast_mode = (self.world_size == 1) and (bs <= 16)
        last = self.num_layers - 1

        qkv = self.adapter.compute_qkv(blocks[0], h, self, bs)

        for layer_idx in range(self.num_layers):
            block = blocks[layer_idx]
            attn_out = self.adapter.attention(qkv, block, layer_idx, bs, self, cache_manager, block_table[:bs])
            attn_out = all_reduce(attn_out)
            mlp_out, res = self.adapter.compute_ffn(block, attn_out, h, self, bs, fast_mode)
            mlp_out = all_reduce(mlp_out)

            if layer_idx < last:
                next_block = blocks[layer_idx + 1]
                qkv, h = self.adapter.compute_next_qkv(next_block, mlp_out, res, self, bs)
            else:
                h = mlp_out + res

        h = self.adapter.final_norm(self.model)(h)
        return self.adapter.lm_head(self.model)(h)

    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._is_graph_ready: return

        is_deepseek = (self.adapter.model_type == "deepseek")
        logger.info("🎯 开始捕获 CUDA Graph ...")

        if batch_sizes:
            _block0 = self.adapter.blocks(self.model)[0]
            _gu, _d = getattr(_block0.mlp, "_gu", None), getattr(_block0.mlp, "_d", None)
            if _gu is not None and _d is not None:
                with torch.no_grad():
                    for bs in batch_sizes:
                        _x = torch.randn(bs, self.hidden_dim, dtype=self.dtype, device=self.device)
                        for _ in range(3): _ = self._fast_mlp(_x, _gu, _d)
                torch.cuda.synchronize()

        # capture/warmup 需要合法 cache 状态：block_table 指向有效 block、seqlens > 0。
        # _block_table_buffer 初始化为 -1（非法 block_id），若 attention 在 seqlens>0 时读它
        # 会 illegal access；且 graph 必须绑定这个【常驻】张量（replay 时框架往里写真实表，
        # graph 读的就是它），不能用临时全 0 buffer。故 capture 前临时填合法 block id，
        # capture 完恢复原值。所有架构（Qwen/DeepSeek）统一走这条路。
        bt_buf = cache_manager._block_table_buffer
        sl_buf = cache_manager._cache_seqlens_buffer
        saved_bt = bt_buf[:self.max_bs].clone()
        saved_sl = sl_buf[:self.max_bs].clone()
        # 每 seq 占前 n_blk 个 block（block_id = i*n_blk .. (i+1)*n_blk-1），seqlens=8。
        # 8 任意取（1..桶上界皆可），只要让 attention 真有 key 可读、且指向合法 block。
        n_blk = (8 + cache_manager.block_size - 1) // cache_manager.block_size

        # DeepSeek: 固定 max_len 桶以进 graph（消除 attention 的 .item() 同步）。
        # 桶上界取 max_blocks*block_size 与 rotary max_position 的较小值，并限到 1024（覆盖常见对话）。
        if is_deepseek:
            self._ds_graph_maxlen = min(1024, self.attention.max_blocks * 256,
                                        self.attention.rotary_emb.cos_cache.shape[2])
            # 运行时标志：attention() 据此选择 graph(varlen+固定桶) / eager(真实 max_len) 路径。
            # capture/warmup 期间恒为 True；forward 时按是否 replay 设定。
            self._use_graph_bucket = True

        for bs in batch_sizes:
            g = torch.cuda.CUDAGraph()
            dummy = torch.randint(0, self.vocab_size, (bs,), device=self.device)
            # 构造合法 cache 状态：每 seq 用 block i*n_blk..(i+1)*n_blk-1，seqlens=8
            for i in range(bs):
                bt_buf[i, :n_blk] = torch.arange(i * n_blk, (i + 1) * n_blk, dtype=torch.int32, device=self.device)
            sl_buf[:bs] = 8
            # Warmup + Capture 都用常驻 bt_buf（已填合法 block id）
            for _ in range(3):
                with torch.no_grad(): self.decode(dummy, bs, cache_manager, bt_buf)
            torch.cuda.synchronize()
            with torch.no_grad(), torch.cuda.graph(g):
                self._logits[:bs] = self.decode(self._input_ids[:bs], bs, cache_manager, bt_buf)
            self._graphs[bs] = g
            logger.info(f"   - Batch size {bs} OK")

        # 恢复 buffer 原值（-1 / 0），避免污染后续真实推理的首步
        bt_buf[:self.max_bs] = saved_bt
        sl_buf[:self.max_bs] = saved_sl

        self._is_graph_ready = True

    def forward(self, input_ids: torch.Tensor | None, cache_manager, batch_size: int) -> torch.Tensor:
        if input_ids is not None:
            # 普通路径：H2D copy
            self._input_ids[:batch_size] = input_ids
        # input_ids=None 表示 _input_ids 已由上一步 GPU→GPU copy 预填充，直接 replay

        is_deepseek = (self.adapter.model_type == "deepseek")
        if is_deepseek:
            # DeepSeek graph 桶有固定 max_len 上界。若任一 seq 实际长度超过桶，replay 会截断有效 key →
            # 必须回退 eager（attention 用真实 max_len）。同步取 max 只发生在回退分支，不影响 replay 路径。
            bucket = getattr(self, "_ds_graph_maxlen", None)
            if bucket is not None and batch_size in self._graphs:
                cur_max = int(cache_manager._cache_seqlens_buffer[:batch_size].max().item())
                use_graph = (cur_max <= bucket)
            else:
                use_graph = False
        else:
            use_graph = batch_size in self._graphs

        if not use_graph:
            # eager 路径（未捕获的 batch_size，或 DeepSeek 序列超 graph 桶上界）：
            # 必须用 cache_manager 已建好的真实 block_table_buffer，
            # 而非全零 buffer——否则多序列会全部读到 block 0 的 KV，互相污染。
            if is_deepseek:
                self._use_graph_bucket = False
            return self.decode(self._input_ids[:batch_size], batch_size, cache_manager,
                               cache_manager._block_table_buffer)
        if is_deepseek:
            self._use_graph_bucket = True
        self._graphs[batch_size].replay()
        return self._logits[:batch_size]
