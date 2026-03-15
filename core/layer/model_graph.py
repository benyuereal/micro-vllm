import logging
import torch
import torch.nn.functional as F
from typing import Dict, List

from core.paged_attention import PagedAttention
from kernel.rmsnorm_add import rmsnorm_
from kernel.rmsnorm_residual import rmsnorm_residual_gemm
from kernel.swiglu import swiglu_gemm
from .rope import RoPE
from core.parallel_config import get_rank, get_world_size, all_reduce
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

        # 通用模块
        self.attention = PagedAttention(num_heads, head_size, kv_num_heads, device, max_batch_size)
        self.rope = RoPE()

        # 【关键恢复】保留 _compiled_mlp 作为公共接口
        self._fast_mlp = self._compile_fn(self._mlp)

        # 初始化
        self.prepare_weights()
        self._alloc_bufs()

        # CUDA Graph
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._is_graph_ready = False

    def _compile_fn(self, fn):
        """保留编译函数供其他类使用"""
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
        """保留纯MLP静态方法供其他类使用"""
        gate_up = x @ gu_weight
        up, gate = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return activated @ d_weight

    # ==========================================
    # 初始化 (Init)
    # ==========================================

    def prepare_weights(self):
        """预缓存转置权重 & TP切分"""
        cfg = self.model.config
        global_num_heads = cfg.num_attention_heads
        global_kv_heads = getattr(cfg, "num_key_value_heads", global_num_heads)
        q_dim = global_num_heads * self.head_size
        kv_dim = global_kv_heads * self.head_size

        for block in self.model.transformer.h:
            # MLP
            w1, w2 = block.mlp.w1.weight, block.mlp.w2.weight
            block.mlp._gu = torch.cat([
                w1.chunk(self.world_size, dim=0)[self.rank],
                w2.chunk(self.world_size, dim=0)[self.rank]
            ], dim=0).t().contiguous()
            block.mlp._d = block.mlp.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # Attn O
            block.attn._o = block.attn.c_proj.weight.chunk(self.world_size, dim=1)[self.rank].t().contiguous()

            # QKV
            w_qkv, b_qkv = block.attn.c_attn.weight, block.attn.c_attn.bias
            w_q, w_k, w_v = w_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
            local_qkv = [w.chunk(self.world_size, dim=0)[self.rank] for w in (w_q, w_k, w_v)]
            block.attn._qkv_w = torch.cat(local_qkv, dim=0).t().contiguous()
            block.attn._qkv_b = None
            if b_qkv is not None:
                b_q, b_k, b_v = b_qkv.split([q_dim, kv_dim, kv_dim], dim=0)
                local_b = [b.chunk(self.world_size, dim=0)[self.rank] for b in (b_q, b_k, b_v)]
                block.attn._qkv_b = torch.cat(local_b, dim=0)

        logger.info(f"✅ 权重预缓存完成 (Rank {self.rank})")

    def _alloc_bufs(self):
        max_b = self.max_bs
        self._input_ids = torch.empty(max_b, dtype=torch.long, device=self.device)
        self._logits = torch.empty(max_b, self.vocab_size, dtype=self.dtype, device=self.device)

        _b = self.model.transformer.h[0]
        qkv_dim = _b.attn._qkv_w.shape[1]
        o_dim = _b.attn._o.shape[1]

        self._normed = torch.empty((max_b, self.hidden_dim), dtype=self.dtype, device=self.device)
        self._qkv = torch.empty(max_b, qkv_dim, dtype=self.dtype, device=self.device)
        self._attn_out = torch.empty(max_b, o_dim, dtype=self.dtype, device=self.device)
        self._residual = torch.empty((max_b, self.hidden_dim), dtype=self.dtype, device=self.device)
        self._gate_up = torch.empty((max_b, self.intermediate_size), dtype=self.dtype, device=self.device)
        self._swiglu_out = torch.empty((max_b, self.intermediate_size // 2), dtype=self.dtype, device=self.device)
        self._mlp_out = torch.empty((max_b, self.hidden_dim), dtype=self.dtype, device=self.device)

    # ==========================================
    # 工具函数 (Utils)
    # ==========================================

    def _qkv_proj(self, h, block, bs):
        """QKV投影"""
        rmsnorm_(h, block.ln_1.weight, self._normed[:bs], block.ln_1.eps)
        qkv_buf = self._qkv[:bs]
        torch.matmul(self._normed[:bs], block.attn._qkv_w, out=qkv_buf)
        if block.attn._qkv_b is not None:
            qkv_buf.add_(block.attn._qkv_b)
        return qkv_buf

    def _flash_attn(self, qkv, block, layer_idx, bs, cache_manager, use_graph_cache):
        """Flash Attention"""
        q, k, v = qkv.reshape(bs, 3, self.num_heads, self.head_size).unbind(dim=1)
        k_cache, v_cache = cache_manager.get(layer_idx)
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]
        block_table = (
            cache_manager._block_table_buffer[:bs]
            if use_graph_cache else
            torch.zeros(bs, self.attention.max_blocks, dtype=torch.int32, device=self.device)
        )

        attn = flash_attn_with_kvcache(
            q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
            k=k.unsqueeze(1), v=v.unsqueeze(1),
            rotary_cos=self.attention._cos_pool, rotary_sin=self.attention._sin_pool,
            cache_seqlens=cache_lens, block_table=block_table,
            causal=True, window_size=(-1, -1), rotary_interleaved=False,
            alibi_slopes=None, num_splits=max(1, 32 // max(1, bs * 4))
        ).squeeze(1)

        out_buf = self._attn_out[:bs]
        torch.matmul(attn.reshape(bs, -1), block.attn._o, out=out_buf)
        return out_buf

    def _ffn(self, x_normed, block, bs, fast_mode=False):
        """FFN计算
        【关键修改】支持动态切换编译/手写路径
        """
        if fast_mode:
            # 小Batch：用编译好的MLP
            return self._fast_mlp(x_normed, block.mlp._gu, block.mlp._d)
        else:
            # 大Batch：用手写Kernel
            torch.matmul(x_normed, block.mlp._gu, out=self._gate_up[:bs])
            swiglu_gemm(self._gate_up[:bs], self._swiglu_out[:bs])
            torch.matmul(self._swiglu_out[:bs], block.mlp._d, out=self._mlp_out[:bs])
            return self._mlp_out[:bs]

    # ==========================================
    # 主推理逻辑 (Main Decode)
    # ==========================================

    def decode(self, input_ids, bs, cache_manager, use_graph_cache):
        """标准Transformer计算流
        【关键修改】根据Batch大小决定是否用 _compiled_mlp
        """
        h = self.model.transformer.wte(input_ids)

        # 【关键修改】小Batch用编译MLP，大Batch用手写Kernel
        fast_mode = (self.world_size == 1) and (bs <= 16)

        # --- 1. 所有层统一循环 ---
        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]

            # Step 1: QKV投影
            qkv = self._qkv_proj(h, block, bs)

            # Step 2: Flash Attention
            attn_out = self._flash_attn(qkv, block, layer_idx, bs, cache_manager, use_graph_cache)
            attn_out = all_reduce(attn_out)

            # Step 3: FFN (包含残差连接)
            rmsnorm_residual_gemm(
                attn_out, h, block.ln_2.weight,
                self._normed[:bs], self._residual[:bs], block.ln_2.eps
            )
            # 【关键修改】传入 use_compiled 参数
            mlp_out = self._ffn(self._normed[:bs], block, bs, fast_mode=fast_mode)
            h = all_reduce(mlp_out).add_(self._residual[:bs])

        # --- 2. Final Norm & Head ---
        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)

    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._is_graph_ready: return

        logger.info("🎯 开始捕获 CUDA Graph ...")

        if batch_sizes:
            _block0 = self.model.transformer.h[0]
            _gu, _d = _block0.mlp._gu, _block0.mlp._d
            with torch.no_grad():
                for bs in batch_sizes:
                    _x = torch.randn(bs, self.hidden_dim, dtype=self.dtype, device=self.device)
                    for _ in range(3): _ = self._fast_mlp(_x, _gu, _d)
            torch.cuda.synchronize()

        # 捕获 Graph
        for bs in batch_sizes:
            g = torch.cuda.CUDAGraph()
            dummy = torch.randint(0, self.vocab_size, (bs,), device=self.device)
            # Warmup
            for _ in range(3):
                with torch.no_grad(): self.decode(dummy, bs, cache_manager, use_graph_cache=False)
            torch.cuda.synchronize()
            # Capture
            with torch.no_grad(), torch.cuda.graph(g):
                self._logits[:bs] = self.decode(self._input_ids[:bs], bs, cache_manager, use_graph_cache=True)
            self._graphs[bs] = g
            logger.info(f"   - Batch size {bs} OK")

        self._is_graph_ready = True

    def forward(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        self._input_ids[:batch_size] = input_ids
        if batch_size not in self._graphs:
            return self.decode(input_ids, batch_size, cache_manager, use_graph_cache=False)
        self._graphs[batch_size].replay()
        return self._logits[:batch_size]