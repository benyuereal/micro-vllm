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

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelGraphRunner:
    """
    📌 修复 KV Cache 形状匹配问题
    """
    
    def __init__(self, model, num_layers: int, num_heads: int, head_size: int,
                 kv_num_heads: int, hidden_dim: int, intermediate_size: int,
                 device: str, max_batch_size: int = 32, dtype: torch.dtype = torch.bfloat16):
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
        """预缓存转置权重，避免显存碎片和重复计算"""
        for idx, block in enumerate(self.model.transformer.h):
            mlp = block.mlp
            
            # MLP: Gate + Up 拼接
            gate_up = torch.cat([mlp.w1.weight, mlp.w2.weight], dim=0)
            mlp._gu = gate_up.t().contiguous()
            
            # MLP: Down
            mlp._d = mlp.c_proj.weight.t().contiguous()
            
            # Attention: Output
            block.attn._o = block.attn.c_proj.weight.t().contiguous()
            
            # Attention: QKV
            block.attn._qkv_w = block.attn.c_attn.weight.t().contiguous()
            block.attn._qkv_b = block.attn.c_attn.bias
        
        logger.info("✅ 权重预缓存完成")
    
    def _allocate_buffers(self):
        """分配 CUDA Graph 所需的静态内存池"""
        max_b = self.max_batch_size
        
        # 输入缓冲区: [Batch]
        self._input_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
        
        # 输出缓冲区: [Batch]
        self._output_ids = torch.empty((max_b,), dtype=torch.long, device=self.device)
    
    def _forward(self, input_ids, batch_size: int, cache_manager, use_graph_cache: bool):
        """
        核心前向逻辑
        """
        # 1. Embedding Lookup
        h = self.model.transformer.wte(input_ids)
        
        # 2. 通过所有 Transformer Layer
        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            
            # --- 加载权重 ---
            w_qkv, b_qkv = block.attn._qkv_w, block.attn._qkv_b
            w_o = block.attn._o
            w_gu, w_d = block.mlp._gu, block.mlp._d
            
            # --- 获取 KV Cache 指针 ---
            if use_graph_cache:
                k_cache, v_cache = cache_manager.get(layer_idx)
                cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]
                block_table = cache_manager._block_table_buffer[:batch_size]
            else:
                # Warmup 模式：获取真实指针但使用假的 Metadata
                k_cache, v_cache = cache_manager.get(layer_idx)
                cache_seqlens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
                block_table = torch.zeros(batch_size, self.attention.max_blocks, dtype=torch.int32, device=self.device)
            
            # --- Attention Block ---
            normed = F.rms_norm(h, h.shape[-1:], block.ln_1.weight, block.ln_1.eps)
            qkv = normed @ w_qkv
            if b_qkv is not None:
                qkv = qkv + b_qkv
            
            qkv_reshaped = qkv.reshape(batch_size, 3, self.num_heads, self.head_size)
            q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]
            
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1),
                k_cache=k_cache,
                v_cache=v_cache,
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                rotary_cos=self.attention._cos_pool,
                rotary_sin=self.attention._sin_pool,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                causal=True,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
            ).squeeze(1)
            
            # --- MLP Block ---
            out = attn.reshape(batch_size, -1) @ w_o
            residual = out + h
            
            normed = F.rms_norm(residual, residual.shape[-1:], block.ln_2.weight, block.ln_2.eps)
            gate_up = normed @ w_gu
            
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            
            mlp_out = activated @ w_d
            h = mlp_out + residual
            
        # 3. 最终归一化 + LM Head
        h = self.model.transformer.ln_f(h)
        logits = self.model.lm_head(h)
        
        # 4. Greedy 采样
        output_ids = torch.argmax(logits, dim=-1)
        
        return output_ids
    
    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]):
        """为指定的 Batch Size 捕获 CUDA Graph"""
        if self._ready:
            return
            
        logger.info(f"🎯 开始捕获 CUDA Graph (Batch sizes: {batch_sizes})")
        
        for bs in batch_sizes:
            self._capture_single(bs, cache_manager)
        
        self._ready = True
        logger.info("✅ 所有 CUDA Graph 捕获完成")
    
    def _capture_single(self, batch_size: int, cache_manager):
        """捕获单个 Batch Size 的 Graph"""
        g = torch.cuda.CUDAGraph()
        
        # 1. 预热 (Warmup)
        self._warmup(batch_size, cache_manager)
        
        # 2. 开始捕获
        with torch.cuda.graph(g):
            # 切片出当前 batch size 大小的视图
            input_view = self._input_ids[:batch_size]
            
            # 执行前向
            output_view = self._forward(input_view, batch_size, cache_manager, use_graph_cache=True)
            
            # 写入输出
            self._output_ids[:batch_size] = output_view
        
        self._graphs[batch_size] = g
        logger.info(f"   - Batch size {batch_size} 捕获成功")
    
    def _warmup(self, batch_size: int, cache_manager, num_warmup: int = 3):
        """预热：让 CUDA 分配好物理内存"""
        dummy_ids = torch.randint(
            0, self.vocab_size, (batch_size,),
            dtype=torch.long, device=self.device
        )
        for _ in range(num_warmup):
            with torch.no_grad():
                self._forward(dummy_ids, batch_size, cache_manager, use_graph_cache=False)
        torch.cuda.synchronize()
    
    def forward(self, input_ids: torch.Tensor, cache_manager, batch_size: int) -> torch.Tensor:
        """
        执行推理
        Args:
            input_ids: [batch_size] 输入的上一个 token
        Returns:
            [batch_size] 预测的下一个 token
        """
        # 拷贝输入到静态内存
        self._input_ids[:batch_size] = input_ids
        
        # 回放 Graph
        if batch_size not in self._graphs:
            # 回退到 Eager 模式
            return self._forward(input_ids, batch_size, cache_manager, use_graph_cache=False)
        
        self._graphs[batch_size].replay()
        
        # 返回结果
        return self._output_ids[:batch_size]