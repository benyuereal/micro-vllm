"""
===================================================================
ModelGraphRunner - 极简全流程 CUDA Graph (Token -> Token)
===================================================================
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List
from core.paged_attention import PagedAttention
from kernel.swiglu import swiglu_fused as swiglu
from .sampler import Sampler  # 引入独立的 Sampler
from kernel.rmsnorm_add import rmsnorm_residual_fused ,rmsnorm

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelGraphRunner:
    """
    输入: Token IDs, Temperatures, Top_Ps
    输出: Next Token IDs
    """
    
    def __init__(self, model, num_layers: int, num_heads: int, head_size: int,
                 kv_num_heads: int, hidden_dim: int, intermediate_size: int,
                 device: str, max_batch_size: int = 32, dtype: torch.dtype = torch.bfloat16,
                 top_k: int = 1000):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = model.config.vocab_size
        self.device = device
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.top_k = top_k
        # 初始化组件
        self.attention = PagedAttention(
            num_heads=num_heads, head_size=head_size, kv_num_heads=kv_num_heads,
            device=device, max_batch_size=max_batch_size
        )
        self.sampler = Sampler()
        
        self.prepare()
        self._allocate_buffers()
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._ready = False
    
    def prepare(self):
        """预缓存转置权重"""
        for idx, block in enumerate(self.model.transformer.h):
            mlp = block.mlp
            gate_up = torch.cat([mlp.w1.weight, mlp.w2.weight], dim=0)
            mlp._gu = gate_up.t().contiguous()
            mlp._d = mlp.c_proj.weight.t().contiguous()
            block.attn._o = block.attn.c_proj.weight.t().contiguous()
            block.attn._qkv_w = block.attn.c_attn.weight.t().contiguous()
            block.attn._qkv_b = block.attn.c_attn.bias
        logger.info("✅ 权重预缓存完成")
    
    def _allocate_buffers(self):
        """分配静态内存池"""
        max_b = self.max_batch_size
        self._input_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
        self._temps = torch.empty((max_b,), dtype=self.dtype, device=self.device)
        self._topps = torch.empty((max_b,), dtype=self.dtype, device=self.device)
        self._output_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
    
    def _forward(self, input_ids, temps, topps, batch_size: int, cache_manager, use_graph_cache: bool):
        """
        核心前向逻辑
        """
        # 1. Embedding
        h = self.model.transformer.wte(input_ids)
        
        # 2. Transformer Layers
        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            
            w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
            w_o = block.attn._o
            w_gu, w_d = block.mlp._gu, block.mlp._d
            
            if use_graph_cache:
                k_cache, v_cache = cache_manager.get(layer_idx)
                cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]
                block_table = cache_manager._block_table_buffer[:batch_size]
            else:
                k_cache, v_cache = cache_manager.get(layer_idx)
                cache_seqlens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
                block_table = torch.zeros(batch_size, self.attention.max_blocks, dtype=torch.int32, device=self.device)
                block_table[:, 0] = 0 # 安全修复
            
            # Attention
            normed = rmsnorm(h, block.ln_1.weight, block.ln_1.eps)
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None: qkv = qkv + b_qkv
            qkv_reshaped = qkv.reshape(batch_size, 3, self.num_heads, self.head_size)
            q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]
            
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
                k=k.unsqueeze(1), v=v.unsqueeze(1),
                rotary_cos=self.attention._cos_pool, rotary_sin=self.attention._sin_pool,
                cache_seqlens=cache_seqlens, block_table=block_table,
                causal=True, window_size=(-1, -1), rotary_interleaved=False, alibi_slopes=None,
            ).squeeze(1)
            
            # MLP
            out = torch.matmul(attn.reshape(batch_size, -1), w_o)
            normed, residual = rmsnorm_residual_fused(
                                x=out,
                                residual=h,
                                weight=block.ln_2.weight,
                                eps=block.ln_2.eps
                            )
            gate_up = torch.matmul(normed, w_gu)
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            mlp_out = torch.matmul(activated, w_d)
            h = mlp_out + residual
            
        # 3. Final Norm + LM Head
        h = self.model.transformer.ln_f(h)
        logits = self.model.lm_head(h)
        
        # 4. 采样 (调用独立模块)
        output_ids = self.sampler._sample_impl(logits.float(), temps, topps, self.top_k)
        
        return output_ids
    
    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._ready: return
        logger.info(f"🎯 开始捕获 CUDA Graph ...")
        for bs in batch_sizes:
            self._capture_single(bs, cache_manager)
        self._ready = True
        logger.info("✅ 捕获完成")
    
    def _capture_single(self, batch_size: int, cache_manager):
        g = torch.cuda.CUDAGraph()
        self._warmup(batch_size, cache_manager)
        
        with torch.cuda.graph(g):
            in_ids = self._input_ids[:batch_size]
            in_temps = self._temps[:batch_size]
            in_topps = self._topps[:batch_size]
            out_ids = self._forward(in_ids, in_temps, in_topps, batch_size, cache_manager, use_graph_cache=True)
            self._output_ids[:batch_size] = out_ids
        
        self._graphs[batch_size] = g
        logger.info(f"   - Batch size {batch_size} OK")
    
    def _warmup(self, batch_size: int, cache_manager, num_warmup: int = 3):
        dummy_ids = torch.randint(0, self.vocab_size, (batch_size,), dtype=torch.long, device=self.device)
        dummy_temps = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        dummy_topps = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        for _ in range(num_warmup):
            with torch.no_grad():
                self._forward(dummy_ids, dummy_temps, dummy_topps, batch_size, cache_manager, use_graph_cache=False)
        torch.cuda.synchronize()
    
    def forward(self, input_ids: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor,
                cache_manager, batch_size: int) -> torch.Tensor:
        self._input_ids[:batch_size] = input_ids
        self._temps[:batch_size] = temperatures.to(self.dtype)
        self._topps[:batch_size] = top_ps.to(self.dtype)
        
        if batch_size not in self._graphs:
            return self._forward(input_ids, temperatures, top_ps, batch_size, cache_manager, use_graph_cache=False)
        
        self._graphs[batch_size].replay()
        return self._output_ids[:batch_size]