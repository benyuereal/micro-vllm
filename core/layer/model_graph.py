import logging
import os
import torch
from typing import Dict, List

from core.paged_attention import PagedAttention
from kernel.rmsnorm import rmsnorm_, rmsnorm_residual_gemm as rmsnorm_residual
from kernel.rotary import compute_slot_mapping
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
                 top_k: int = 1000, max_context_length: int = 1024):
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
        # final_norm 权重 + eps：缓存以用融合 rmsnorm_ 替代 HF 原生 RMSNorm
        # （HF 原生 .float()/.to(dtype) 产生 2 个 D2D copy，bs=512 时 ~410us/step）。
        _fn = self.adapter.final_norm(self.model)
        self._final_norm_w = _fn.weight.data
        self._final_norm_eps = getattr(_fn, "variance_epsilon", getattr(_fn, "eps", 1e-6))

        # 通用模块
        # PagedAttention 的 head 维度 = KV cache 存储维度（GQA=head_size, MLA=latent_dim）
        # rope_dim = RoPE 实际作用维度（GQA=head_size, MLA=qk_rope_head_dim）
        # cos/sin pool 长度需覆盖 max_context_length（Qwen3 用此池做 RoPE）。
        block_size = 256
        max_blocks_for_pool = (max_context_length + block_size - 1) // block_size
        self.attention = PagedAttention(num_heads, head_size, kv_num_heads, device, max_batch_size,
                                        max_blocks=max_blocks_for_pool,
                                        rope_dim=self.adapter.rope_dim(model.config),
                                        rope_theta=self.adapter.rope_theta(model.config))
        self.rope = RoPE()

        # 初始化
        self.adapter.prepare_weights(self.model, self.world_size, self.rank)
        self._alloc_bufs()

        # CUDA Graph：两架构统一固定 1024 上下文，graph 选择键 (bs, None)，无选桶/无 eager
        # 回退/无运行期 .item() 同步。DeepSeek 把分页 KV gather 成 [bs, max_len, 576]，max_len
        # 进张量形状故需固定 1024（越界 key 由 cu_seqlens_k 截断）；Qwen 用 flash_attn_with_kvcache
        # (block_table=...)，无 seq_len 维不读 max_len。>1024 长序列留待 tile op 恢复。
        self._deepseek_fixed_maxlen = 1024 if self.adapter.model_type == "deepseek" else None
        self._graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}
        self._is_graph_ready = False
        # replay 前 forward/capture 设置：DeepSeek=1024，Qwen=None（attention 不读）。
        self._cur_bucket_maxlen = None

    def _alloc_bufs(self):
        max_b = self.max_bs
        self._input_ids = torch.empty(max_b, dtype=torch.long, device=self.device)
        self._logits = torch.empty(max_b, self.vocab_size, dtype=self.dtype, device=self.device)
        # 最终 hidden（final_norm 输出，lm_head 之前）。graph 捕获到此为止，lm_head 在
        # replay 后 eager 跑——避免把 lm_head 的 [bs,vocab] 输出 copy 进 graph buffer
        # （bs=512 vocab=151936 bf16 = 155MB D2D copy，profiled 409us/step，是落后 nano
        # 的 0.41ms gap 主因之一）。hidden 仅 [bs, hidden]=1MB，copy 可忽略。
        self._hidden = torch.empty(max_b, self.hidden_dim, dtype=self.dtype, device=self.device)
        # 当前步各 seq 写入 paged KV 的 slot（prerope+store 路径用）。每步 decode 开头算一次。
        self._slot_mapping = torch.empty(max_b, dtype=torch.int32, device=self.device)
        # flash 读取长度 = cache_seqlens + 1（含当前 token）。prerope 路径专用。
        self._flash_seqlens = torch.empty(max_b, dtype=torch.int32, device=self.device)

        # 由 adapter 决定 buffer 形状（不同架构需要不同的中间张量）；统一挂到 self 上。
        # 架构无关的 key（_h_buf/_qkv/_attn_out/_residual）各 adapter 必返回，
        # 架构专属的（_x16/_absorb_idx/_cos_full/_sin_full，仅 DeepSeek）缺省为 None。
        bufs = self.adapter.alloc_bufs(self.model, max_b, self.hidden_dim, self.dtype, self.device)
        for name in ("_h_buf", "_qkv", "_attn_out", "_residual",
                     "_x16", "_absorb_idx", "_cos_full", "_sin_full"):
            setattr(self, name, bufs.get(name))

    # ==========================================
    # 主推理逻辑
    # ==========================================

    def decode(self, input_ids, bs, cache_manager, block_table):
        embed = self.adapter.embed(self.model)
        blocks = self.adapter.blocks(self.model)
        h = embed(input_ids)
        last = self.num_layers - 1

        # prerope+store 路径：每步算一次 slot_mapping（当前 token 写入位置），
        # 供各层 store_kvcache 用。GPU 原地算，graph 友好。
        # 同时算 flash_seqlens = cache_seqlens + 1（含当前 token，flash 读取长度）：
        # micro 的 cache_seqlens 在 commit() 里 +1（forward 后），故 forward 内是旧值，
        # 需 +1 让 flash 读到刚 store 的当前 token（对齐 nano：context_lens 是新长度）。
        if getattr(self.adapter, "use_prerope_decode", False):
            compute_slot_mapping(block_table, cache_manager._cache_seqlens_buffer[:bs],
                                 cache_manager.block_size, self._slot_mapping[:bs])
            torch.add(cache_manager._cache_seqlens_buffer[:bs], 1, out=self._flash_seqlens[:bs])

        qkv = self.adapter.compute_qkv(blocks[0], h, self, bs)

        for layer_idx in range(self.num_layers):
            block = blocks[layer_idx]
            attn_out = self.adapter.attention(qkv, block, layer_idx, bs, self, cache_manager, block_table[:bs])
            attn_out = all_reduce(attn_out)
            mlp_out, res = self.adapter.compute_ffn(block, attn_out, h, self, bs)
            mlp_out = all_reduce(mlp_out)

            if layer_idx < last:
                next_block = blocks[layer_idx + 1]
                qkv, h = self.adapter.compute_next_qkv(next_block, mlp_out, res, self, bs)
            else:
                h = mlp_out + res

        # final_norm 用融合 rmsnorm_ 直写 _hidden（省 HF 原生 RMSNorm 的 2 个 bf16↔fp32 D2D copy，
        # 且省 _hidden[:bs]=h 的 1MB copy）。图捕获时 _hidden[:bs] 即此 kernel 输出，无需额外赋值。
        rmsnorm_(h, self._final_norm_w, self._hidden[:bs], self._final_norm_eps)
        return self._hidden[:bs]  # lm_head 移出 graph（见 forward），避免 155MB logits D2D copy

    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._is_graph_ready: return

        logger.info("🎯 开始捕获 CUDA Graph ...")

        # capture/warmup 需要合法 cache 状态：block_table 指向有效 block、seqlens > 0。
        # _block_table_buffer 初始化为 -1（非法 block_id），若 attention 在 seqlens>0 时读它
        # 会 illegal access；且 graph 必须绑定这个【常驻】张量（replay 时框架往里写真实表，
        # graph 读的就是它），不能用临时全 0 buffer。故 capture 前临时填合法 block id，
        # capture 完恢复原值。所有架构（Qwen/DeepSeek）统一走这条路。
        bt_buf = cache_manager._block_table_buffer
        sl_buf = cache_manager._cache_seqlens_buffer
        saved_bt = bt_buf[:self.max_bs].clone()
        saved_sl = sl_buf[:self.max_bs].clone()

        # warmup：block_table 前 n_blk_warmup 列填 block 0..n-1（结构合法即可，数值无意义）。
        # 两架构统一按固定 1024 算 n_blk_warmup=ceil(1024/256)=4；Qwen 不 gather 多填无害。
        block_size = cache_manager.block_size
        fixed_maxlen = self._deepseek_fixed_maxlen if self._deepseek_fixed_maxlen is not None \
            else cache_manager._block_table_buffer.shape[1] * block_size
        n_blk_warmup = (fixed_maxlen + block_size - 1) // block_size  # 1024/256=4
        # 启动期约束：列数 max_seq_blocks 必须 ≥ n_blk_warmup，否则 arange(fixed_maxlen)//block_size
        # 读越界列。满足后运行期无需 .item() 同步检查。
        max_seq_blocks = cache_manager._block_table_buffer.shape[1]
        assert max_seq_blocks >= n_blk_warmup, \
            f"block_table 列数 {max_seq_blocks} 不足以支撑固定 max_len " \
            f"{fixed_maxlen}（需 ≥{n_blk_warmup}，即 max_tokens ≥ fixed_maxlen）"
        assert cache_manager.n_blocks >= n_blk_warmup, \
            f"warmup 需 {n_blk_warmup} 个 block，cache 只有 {cache_manager.n_blocks}"

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
                self.decode(self._input_ids[:bs], bs, cache_manager, bt_buf)  # 直写 _hidden
            self._graphs[(bs, None)] = g
            logger.info(f"   - Batch size {bs} OK")

        # 恢复 buffer 原值（-1 / 0），避免污染后续真实推理的首步
        bt_buf[:self.max_bs] = saved_bt
        sl_buf[:self.max_bs] = saved_sl

        self._is_graph_ready = True

    def _compute_logits(self, bs) -> torch.Tensor:
        """lm_head：hidden → logits。在 graph 外 eager 跑（graph 只捕获到 hidden）。

        对齐 nano-vllm：lm_head 是 [bs,vocab] 大 GEMM，输出 155MB（bs=512）。
        若把它放进 graph，需把输出 copy 进常驻 _logits buffer（D2D 155MB=409us/step）；
        放 graph 外直接 F.linear 返回新 tensor 给 sampler，省掉这次 copy。
        lm_head 本身是单 kernel（1.4ms），graph 外多一次 launch 开销可忽略。"""
        return self.adapter.lm_head(self.model)(self._hidden[:bs])

    def forward(self, input_ids: torch.Tensor | None, cache_manager, batch_size: int) -> torch.Tensor:
        if input_ids is not None:
            # 普通路径：H2D copy
            self._input_ids[:batch_size] = input_ids
        # input_ids=None 表示 _input_ids 已由上一步 GPU→GPU copy 预填充，直接 replay

        # DEBUG eager 路径（不走 graph），用于定位 kernel 正确性
        if os.environ.get("MICRO_EAGER_DECODE"):
            with torch.no_grad():
                self.decode(self._input_ids[:batch_size], batch_size,
                            cache_manager, cache_manager._block_table_buffer)  # 直写 _hidden
            return self._compute_logits(batch_size)

        # 统一选 graph：(bs, None) 无架构分支、无选桶、无 .item() 同步。越界由启动期 assert 保证。
        key = (batch_size, None)
        if key not in self._graphs:
            raise RuntimeError(f"未捕获的 batch_size={batch_size}（请在 capture 的 batch_sizes 中加入）")
        self._graphs[key].replay()
        # lm_head 在 graph 外 eager 跑（省 155MB logits D2D copy）
        return self._compute_logits(batch_size)
