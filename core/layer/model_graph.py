"""
===================================================================
ModelGraphRunner - 支持 Top-P/Top-K 采样的全流程 Graph
===================================================================
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List
from core.paged_attention import PagedAttention
from kernel.swiglu import swiglu_fused as swiglu

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
                 top_k: int = 1000): # 新增：固定 top_k
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
        self.top_k = top_k # 固定下来，因为 Graph 需要静态形状
        self.sample = Sample()
        
        # 初始化 PagedAttention
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device,
            max_batch_size=max_batch_size
        )
        
        # 预缓存权重
        self.prepare()
        
        # 初始化静态缓冲区
        self._allocate_buffers()
        
        # Graph 存储
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
        """分配静态内存池 (新增采样参数缓冲区)"""
        max_b = self.max_batch_size
        
        # 1. 输入 Token [Batch]
        self._input_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
        
        # 2. 输入采样参数 [Batch]
        self._temps = torch.empty((max_b,), dtype=self.dtype, device=self.device)
        self._topps = torch.empty((max_b,), dtype=self.dtype, device=self.device)
        
        # 3. 输出 Token [Batch]
        self._output_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
    
    def _sample_impl(self, logits, temp, top_p, top_k):
        """
        内联你的采样代码
        注意：为了 Graph 捕获，移除了所有注释，保持纯算子流
        """
        logits = logits / temp[:, None]
        
        # Top-K
        vals, idxs = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        
        # Top-P
        sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
        cum_p = torch.cumsum(sorted_p, dim=-1)
        valid = cum_p < top_p[:, None]
        valid[..., 0] = True
        
        probs_masked = sorted_p * valid
        # 注意：这里要处理除零，但在 Graph 里我们可以信任 valid[...,0]=True
        probs_norm = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
        
        # 采样
        samples = torch.multinomial(probs_norm, 1)
        
        # 映射回原索引
        topk_idx = sorted_i.gather(-1, samples)
        return idxs.gather(-1, topk_idx).squeeze(-1)
    
    def _forward(self, input_ids, temps, topps, batch_size: int, cache_manager, use_graph_cache: bool):
        """
        核心前向逻辑 (包含采样)
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
            
            # Attention
            normed = F.rms_norm(h, h.shape[-1:], block.ln_1.weight, block.ln_1.eps)
            qkv = normed @ w_qkv
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
            out = attn.reshape(batch_size, -1) @ w_o
            residual = out + h
            normed = F.rms_norm(residual, residual.shape[-1:], block.ln_2.weight, block.ln_2.eps)
            gate_up = normed @ w_gu
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            mlp_out = activated @ w_d
            h = mlp_out + residual
            
        # 3. Final Norm + LM Head
        h = self.model.transformer.ln_f(h)
        logits = self.model.lm_head(h)
        
        # 4. 采样 (你的代码)
        # 注意：logits 可能是 bf16，转成 fp32 采样更稳定
        output_ids = self._sample_impl(logits.float(), temps, topps, self.top_k)
        
        return output_ids
    
    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        if self._ready: return
        logger.info(f"🎯 开始捕获 CUDA Graph (带 Top-P/Top-K) ...")
        for bs in batch_sizes:
            self._capture_single(bs, cache_manager)
        self._ready = True
        logger.info("✅ 捕获完成")
    
    def _capture_single(self, batch_size: int, cache_manager):
        g = torch.cuda.CUDAGraph()
        
        # 预热
        self._warmup(batch_size, cache_manager)
        
        with torch.cuda.graph(g):
            # 静态视图
            in_ids = self._input_ids[:batch_size]
            in_temps = self._temps[:batch_size]
            in_topps = self._topps[:batch_size]
            
            # 前向
            out_ids = self._forward(in_ids, in_temps, in_topps, batch_size, cache_manager, use_graph_cache=True)
            
            # 输出
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
        """
        Args:
            input_ids: [batch_size]
            temperatures: [batch_size]
            top_ps: [batch_size]
        """
        # 填充输入缓冲区
        self._input_ids[:batch_size] = input_ids
        self._temps[:batch_size] = temperatures.to(self.dtype) # 确保类型一致
        self._topps[:batch_size] = top_ps.to(self.dtype)
        
        # 回放
        if batch_size not in self._graphs:
            return self._forward(input_ids, temperatures, top_ps, batch_size, cache_manager, use_graph_cache=False)
        
        self._graphs[batch_size].replay()
        
        return self._output_ids[:batch_size]