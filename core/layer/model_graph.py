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
        # graph 选择键统一为 (bs, None)——两架构都只按 batch_size 选 graph，无长度分桶。
        # DeepSeek attention 内部把分页 KV gather 成 [bs, max_len, 576]，max_len 进张量形状。
        # 这里固定 max_len=4096（=max_position_embeddings）通吃所有序列长度：越界 key 由
        # flash_attn_varlen_func 的 cu_seqlens_k 截断不参与 attention。一个 graph 形状通吃，
        # 无 if-else、无选桶、无 eager 回退——控制流对齐 v2 engine，且为 tile op 融合提供
        # 编译期固定的形状目标（后续把 attention 内部换成 TileLang 融合 kernel 时外部不动）。
        # 代价：短序列也按 4096 做 gather/kv_b_proj（多算），但 tile op 重构后 gather 这步
        # 会被 tile 化重写，该代价消失。Qwen 的 flash_attn 吃 block_table 本就无 seq_len 维。
        self._deepseek_fixed_maxlen = 4096 if self.adapter.model_type == "deepseek" else None
        self._graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}
        self._is_graph_ready = False
        # replay 前由 forward/capture 设置：DeepSeek=4096（固定），Qwen=None（attention 不读）。
        # attention() 据此取 max_len，不再 .item() 同步。
        self._cur_bucket_maxlen = None

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

        # warmup 填充：所有 seq 的 block_table 前 n_blk_warmup 列指向 block 0..n_blk_warmup-1
        # （共用前几个 block，warmup 数值无意义只需结构合法不越界）。
        # DeepSeek 固定 max_len=4096，attention 内部 arange(4096)//block_size 最大读第 16 列，
        # 故 n_blk_warmup 需 ≥16；Qwen 不 gather，只需 seqlens>0 让 flash_attn 有 key 可读。
        block_size = cache_manager.block_size
        if is_deepseek:
            n_blk_warmup = (self._deepseek_fixed_maxlen + block_size - 1) // block_size  # 4096/256=16
            # 启动期约束：block_table 列数（max_seq_blocks）必须 ≥ n_blk_warmup，否则
            # attention 内部 arange(fixed_maxlen)//block_size 会读越界列。满足此约束后
            # 运行期无需任何 .item() 同步检查（forward 无长度判断）。
            max_seq_blocks = cache_manager._block_table_buffer.shape[1]
            assert max_seq_blocks >= n_blk_warmup, \
                f"block_table 列数 {max_seq_blocks} 不足以支撑固定 max_len " \
                f"{self._deepseek_fixed_maxlen}（需 ≥{n_blk_warmup}，即 max_tokens ≥ fixed_maxlen）"
        else:
            n_blk_warmup = (8 + block_size - 1) // block_size  # 1 个 block 够 warmup
        # 检查 block 数足够（n_blocks=81 ≥ 16，OK）
        assert cache_manager.n_blocks >= n_blk_warmup, \
            f"warmup 需 {n_blk_warmup} 个 block，cache 只有 {cache_manager.n_blocks}"

        # DeepSeek: 固定 max_len=4096 通吃。capture/warmup 期间设 _cur_bucket_maxlen 让
        # attention 走固定 max_len 路径（与 forward replay 一致）。
        if is_deepseek:
            self._cur_bucket_maxlen = self._deepseek_fixed_maxlen

        for bs in batch_sizes:
            g = torch.cuda.CUDAGraph()
            dummy = torch.randint(0, self.vocab_size, (bs,), device=self.device)
            # 所有 seq 共用 block 0..n_blk_warmup-1（结构合法即可，warmup 不验证数值）
            warmup_blocks = torch.arange(n_blk_warmup, dtype=torch.int32, device=self.device)
            for i in range(bs):
                bt_buf[i, :n_blk_warmup] = warmup_blocks
            sl_buf[:bs] = 8
            # Warmup + Capture 都用常驻 bt_buf（已填合法 block id）
            for _ in range(3):
                with torch.no_grad(): self.decode(dummy, bs, cache_manager, bt_buf)
            torch.cuda.synchronize()
            with torch.no_grad(), torch.cuda.graph(g):
                self._logits[:bs] = self.decode(self._input_ids[:bs], bs, cache_manager, bt_buf)
            self._graphs[(bs, None)] = g
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

        # 统一选 graph：两架构都按 (bs, None) 选，无 is_deepseek 分支、无选桶、无 eager 回退、
        # 无运行期 .item() 同步。DeepSeek 固定 max_len=4096 通吃，序列长度不影响 graph 形状；
        # 越界 key 由 cu_seqlens_k 截断。block_table 列越界由启动期 assert 保证（见 __init__）。
        key = (batch_size, None)
        if key not in self._graphs:
            raise RuntimeError(f"未捕获的 batch_size={batch_size}（请在 capture 的 batch_sizes 中加入）")
        self._graphs[key].replay()
        return self._logits[:batch_size]
