"""
===================================================================
ModelGraphRunner - 整个模型层的CUDA Graph封装 (Fixed KV Cache Shape)
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
                 device: str, max_batch_size: int = 16, dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
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
        
        # 初始化缓冲区
        self._allocate_buffers()
        
        # 预缓存权重
        self.prepare()
        
        # Graph存储
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._ready = False
    
    def prepare(self):
        """预缓存转置权重"""
        for idx, block in enumerate(self.model.transformer.h):
            mlp = block.mlp
            
            # Gate + Up
            gate_up = torch.cat([mlp.w1.weight, mlp.w2.weight], dim=0)
            mlp._gu = gate_up.t().contiguous()
            
            # Down
            mlp._d = mlp.c_proj.weight.t().contiguous()
            
            # Attention output
            block.attn._o = block.attn.c_proj.weight.t().contiguous()
            
            # QKV
            block.attn._qkv_w = block.attn.c_attn.weight.t().contiguous()
            block.attn._qkv_b = block.attn.c_attn.bias
        
        logger.info("✅ ModelGraphRunner 权重预缓存完成")
    
    def _allocate_buffers(self):
        """初始化静态缓冲区"""
        max_b = self.max_batch_size
        self._hidden = torch.empty((max_b, self.hidden_dim), dtype=self.dtype, device=self.device)
        self._output = torch.empty_like(self._hidden)
    
    def _forward_pass(self, h, batch_size: int, cache_manager, use_graph_cache: bool = True):
        """
        🔧 核心逻辑：单次前向传播通过所有层
        """
        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            
            # 提取权重
            w_qkv = block.attn._qkv_w
            b_qkv = block.attn._qkv_b
            w_o = block.attn._o
            w_gu = block.mlp._gu
            w_d = block.mlp._d
            
            # 获取 KV Cache
            if use_graph_cache:
                # 真实推理模式
                k_cache, v_cache = cache_manager.get(layer_idx)
                cache_seqlens = cache_manager._cache_seqlens_buffer[:batch_size]
                block_table = cache_manager._block_table_buffer[:batch_size]
            else:
                # Warmup 模式：
                # 🔧 关键修复：获取真实的 KV Cache 指针以保证形状正确
                k_cache, v_cache = cache_manager.get(layer_idx)
                
                # 只将 metadata 设为 dummy
                cache_seqlens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
                block_table = torch.zeros(batch_size, self.attention.max_blocks, dtype=torch.int32, device=self.device)
            
            # === Attention Block ===
            normed = torch.nn.functional.rms_norm(h, h.shape[-1:], block.ln_1.weight, block.ln_1.eps)
            
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None:
                qkv = qkv + b_qkv
            
            # Split QKV
            qkv_reshaped = qkv.reshape(batch_size, 3, self.num_heads, self.head_size)
            q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]
            
            # Flash Attention
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
            
            # === MLP Block ===
            out = torch.matmul(attn.reshape(batch_size, -1), w_o)
            residual = out + h
            
            normed = torch.nn.functional.rms_norm(residual, residual.shape[-1:], block.ln_2.weight, block.ln_2.eps)
            gate_up = torch.matmul(normed, w_gu)
            
            up, gate = gate_up.chunk(2, dim=-1)
            activated = swiglu(gate, up)
            
            mlp_out = torch.matmul(activated, w_d)
            h = mlp_out + residual
            
        return h
    
    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16]):
        """捕获CUDA Graph"""
        if self._ready:
            return
        
        logger.info(f"Capturing ModelGraph for {self.num_layers} layers x {len(batch_sizes)} batch sizes")
        
        for bs in batch_sizes:
            self._capture_single(bs, cache_manager)
        
        self._ready = True
        logger.info("ModelGraph capture completed")
    
    def _capture_single(self, batch_size: int, cache_manager):
        """捕获单个batch_size的graph"""
        g = torch.cuda.CUDAGraph()
        
        # 预热
        self._warmup(batch_size, cache_manager)
        
        with torch.cuda.graph(g):
            # 输入视图
            h = self._hidden[:batch_size]
            
            # 调用核心逻辑
            output_h = self._forward_pass(h, batch_size, cache_manager, use_graph_cache=True)
            
            # 写入输出缓冲区
            self._output[:batch_size] = output_h
        
        self._graphs[batch_size] = g
    
    def _warmup(self, batch_size: int, cache_manager, num_warmup: int = 3):
        """预热"""
        dummy_hidden = torch.randn(
            batch_size, self.hidden_dim,
            dtype=self.dtype, device=self.device
        )
        
        for _ in range(num_warmup):
            with torch.no_grad():
                self._eager(dummy_hidden, batch_size, cache_manager)
        
        torch.cuda.synchronize()

    def _eager(self, hidden_states, batch_size: int, cache_manager):
        """Eager模式的前向（用于预热）"""
        h = hidden_states.squeeze(1) if hidden_states.dim() == 3 else hidden_states
        
        # 调用核心逻辑
        return self._forward_pass(h, batch_size, cache_manager, use_graph_cache=False)
    
    def forward(self, hidden_states, cache_manager, batch_size: int):
        """执行前向传播"""
        # 输入拷贝
        self._hidden[:batch_size] = hidden_states.squeeze(1) if hidden_states.dim() == 3 else hidden_states
        
        # Replay Graph
        if batch_size not in self._graphs:
            print(f"Graph not found for batch_size={batch_size}")
            return self._eager(hidden_states, batch_size, cache_manager)
        
        self._graphs[batch_size].replay()
        
        # 返回输出视图
        return self._output[:batch_size]
    
    @property
    def is_ready(self) -> bool:
        return self._ready