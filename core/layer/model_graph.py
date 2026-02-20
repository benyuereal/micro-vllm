"""
===================================================================
ModelGraphRunner - 整个模型层的CUDA Graph封装
===================================================================

📌 **功能**：
   将所有transformer layer封装到一个CUDA Graph中，一次replay完成所有层的计算

⚡ **性能提升**：
   - 减少 N次 graph replay → 1次 graph replay
   - 消除层间调度overhead
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
    📌 **模型Graph运行器** - 一次前向处理所有层
    
    🔍 **设计**:
        - 将所有transformer layer的计算封装到一个CUDA Graph中
        - 使用静态buffer避免重复内存分配
        - 支持多个batch_size的graph
    
    ⚡ **性能**:
        - 原来：num_layers次 graph.replay()
        - 现在：1次 graph.replay()
    """
    
    def __init__(self, model, num_layers: int, num_heads: int, head_size: int,
                 kv_num_heads: int, hidden_dim: int, intermediate_size: int,
                 device: str, max_batch_size: int = 16):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.device = device
        self.max_batch_size = max_batch_size
        
        # 初始化 PagedAttention
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device,
            max_batch_size=max_batch_size
        )
        
        # 初始化缓冲区
        self._init_buffers()
        
        # 预缓存权重
        self.prepare()
        
        # Graph存储
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}  # batch_size -> graph
        self._ready = False
        
        # 预分配输出缓冲区（避免每次 clone）
        self._output_buffer = torch.empty(
            (self.max_batch_size, self.hidden_dim),
            dtype=torch.bfloat16, device=self.device
        )
    
    def prepare(self):
        """预缓存转置权重（与layer.py一致）"""
        for idx, block in enumerate(self.model.transformer.h):
            mlp = block.mlp
            
            # Gate + Up: [2*intermediate, hidden] -> [hidden, 2*intermediate]
            gate_up = torch.cat([mlp.w1.weight, mlp.w2.weight], dim=0)
            mlp._gu = gate_up.t().contiguous()
            
            # Down: [intermediate, hidden] -> [hidden, intermediate]
            mlp._d = mlp.c_proj.weight.t().contiguous()
            
            # Attention output
            block.attn._o = block.attn.c_proj.weight.t().contiguous()
            
            # QKV
            block.attn._qkv_w = block.attn.c_attn.weight.t().contiguous()
            if block.attn.c_attn.bias is not None:
                block.attn._qkv_b = block.attn.c_attn.bias
            else:
                block.attn._qkv_b = None
        
        logger.info("✅ ModelGraphRunner 权重预缓存完成")
    
    def _init_buffers(self):
        """初始化静态缓冲区"""
        max_b = self.max_batch_size
        
        # 输入缓冲区
        self._hidden = torch.empty(
            (max_b, self.hidden_dim),
            dtype=torch.bfloat16, device=self.device
        )
        
        # 层间临时缓冲区（复用同一个buffer）
        self._intermediate = torch.empty_like(self._hidden)
        
        # 残差缓冲区
        self._residual = torch.empty_like(self._hidden)
        
        # 归一化输出缓冲区
        self._normed_1 = torch.empty_like(self._hidden)
        self._normed_2 = torch.empty_like(self._hidden)
        
        # QKV缓冲区
        self._qkv = torch.empty(
            (max_b, 3 * self.hidden_dim),
            dtype=torch.bfloat16, device=self.device
        )
        
        # MLP中间缓冲区
        self._gate_up = torch.empty(
            (max_b, 2 * self.intermediate_size),
            dtype=torch.bfloat16, device=self.device
        )
        
        # 最终输出缓冲区
        self._output = torch.empty_like(self._hidden)
    
    def capture(self, cache_manager, batch_sizes: List[int] = [1, 2, 4, 8, 16]):
        """
        捕获CUDA Graph
        
        Args:
            cache_manager: KVCacheManager实例
            batch_sizes: 需要捕获的batch_size列表
        """
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
            # ============ Layer 0 ============
            h = self._hidden[:batch_size]
            
            for layer_idx in range(self.num_layers):
                block = self.model.transformer.h[layer_idx]
                
                # 获取权重
                w_qkv = block.attn._qkv_w
                b_qkv = block.attn._qkv_b
                w_o = block.attn._o
                w_gu = block.mlp._gu
                w_d = block.mlp._d
                
                # 获取KV cache
                k_cache, v_cache = cache_manager.get(layer_idx)
                
                # === Attention ===
                # RMSNorm
                normed = torch.nn.functional.rms_norm(
                    h, h.shape[-1:], block.ln_1.weight, block.ln_1.eps
                )
                self._normed_1[:batch_size] = normed
                
                # QKV Linear
                self._qkv[:batch_size] = torch.matmul(self._normed_1[:batch_size], w_qkv)
                if b_qkv is not None:
                    self._qkv[:batch_size] = self._qkv[:batch_size] + b_qkv
                
                # Split QKV
                q = self._qkv[:batch_size][:, :self.hidden_dim].reshape(
                    batch_size, self.num_heads, self.head_size
                )
                k = self._qkv[:batch_size][:, self.hidden_dim:2*self.hidden_dim].reshape(
                    batch_size, self.kv_num_heads, self.head_size
                )
                v = self._qkv[:batch_size][:, 2*self.hidden_dim:].reshape(
                    batch_size, self.kv_num_heads, self.head_size
                )
                
                # FlashAttention with KV cache
                attn = flash_attn_with_kvcache(
                    q=q.unsqueeze(1),
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k=k.unsqueeze(1),
                    v=v.unsqueeze(1),
                    rotary_cos=self.attention._cos_pool,
                    rotary_sin=self.attention._sin_pool,
                    cache_seqlens=cache_manager._cache_seqlens_buffer[:batch_size],
                    block_table=cache_manager._block_table_buffer[:batch_size],
                    causal=True,
                    window_size=(-1, -1),
                    rotary_interleaved=False,
                    alibi_slopes=None,
                ).squeeze(1)
                
                # === MLP ===
                # O Proj
                out = torch.matmul(attn.reshape(batch_size, -1), w_o)
                
                # First Residual
                self._residual[:batch_size] = out + h
                
                # MLP RMSNorm
                normed = torch.nn.functional.rms_norm(
                    self._residual[:batch_size],
                    self._residual[:batch_size].shape[-1:],
                    block.ln_2.weight,
                    block.ln_2.eps
                )
                self._normed_2[:batch_size] = normed
                
                # Gate + Up
                gate_up = torch.matmul(self._normed_2[:batch_size], w_gu)
                
                # SwiGLU
                activated = swiglu(gate_up)

                # Down Proj
                mlp_out = torch.matmul(activated, w_d)
                
                # Second Residual -> 存到intermediate或output
                if layer_idx < self.num_layers - 1:
                    # 不是最后一层，输出给下一层
                    self._intermediate[:batch_size] = mlp_out + self._residual[:batch_size]
                    h = self._intermediate[:batch_size]
                else:
                    # 最后一层
                    self._output[:batch_size] = mlp_out + self._residual[:batch_size]
        
        self._graphs[batch_size] = g
    
    def _warmup(self, batch_size: int, cache_manager, num_warmup: int = 3):
        """预热"""
        dummy_hidden = torch.randn(
            batch_size, self.hidden_dim,
            dtype=torch.bfloat16, device=self.device
        )
        
        for _ in range(num_warmup):
            with torch.no_grad():
                self._eager(dummy_hidden, batch_size, cache_manager)
        
        torch.cuda.synchronize()

    def _eager(self, hidden_states, batch_size: int, cache_manager):
        """Eager模式的前向（用于预热）"""
        h = hidden_states.squeeze(1) if hidden_states.dim() == 3 else hidden_states
        
        for layer_idx in range(self.num_layers):
            block = self.model.transformer.h[layer_idx]
            
            w_qkv = block.attn._qkv_w
            b_qkv = block.attn._qkv_b
            w_o = block.attn._o
            w_gu = block.mlp._gu
            w_d = block.mlp._d
            k_cache, v_cache = cache_manager.get(layer_idx)
            
            # Attention
            normed = torch.nn.functional.rms_norm(
                h, h.shape[-1:], block.ln_1.weight, block.ln_1.eps
            )
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None:
                qkv = qkv + b_qkv
            
            q = qkv[:, :self.hidden_dim].reshape(batch_size, self.num_heads, self.head_size)
            k = qkv[:, self.hidden_dim:2*self.hidden_dim].reshape(batch_size, self.kv_num_heads, self.head_size)
            v = qkv[:, 2*self.hidden_dim:].reshape(batch_size, self.kv_num_heads, self.head_size)
            
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1),
                k_cache=k_cache,
                v_cache=v_cache,
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                rotary_cos=self.attention._cos_pool,
                rotary_sin=self.attention._sin_pool,
                cache_seqlens=torch.ones(batch_size, dtype=torch.int32, device=self.device),
                block_table=torch.zeros(batch_size, self.attention.max_blocks, 
                                    dtype=torch.int32, device=self.device),
                causal=True,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
            ).squeeze(1)
            
            # MLP
            out = torch.matmul(attn.reshape(batch_size, -1), w_o)
            h = out + h
            
            normed = torch.nn.functional.rms_norm(
                h, h.shape[-1:], block.ln_2.weight, block.ln_2.eps
            )
            gate_up = torch.matmul(normed, w_gu)
            activated = swiglu(gate_up)
            mlp_out = torch.matmul(activated, w_d)
            h = mlp_out + h
        
        return h
    
    def forward(self, hidden_states, cache_manager, batch_size: int):
        """
        执行前向传播
        
        Args:
            hidden_states: 输入hidden states [batch_size, hidden_dim]
            cache_manager: KVCacheManager实例
            batch_size: batch大小
            
        Returns:
            output: 输出hidden states [batch_size, hidden_dim]
        """
        # 将输入复制到缓冲区
        self._hidden[:batch_size] = hidden_states.squeeze(1) if hidden_states.dim() == 3 else hidden_states
        
        # Replay Graph
        if batch_size not in self._graphs:
            print(f"Graph not found for batch_size={batch_size}")
            return self._eager(hidden_states, batch_size, cache_manager)
        
        self._graphs[batch_size].replay()
        
        # 返回输出（直接返回 view，避免 clone）
        # 注意：由于是单线程顺序执行，下一个 step 前 hidden_states 已被 norm 处理完
        return self._output[:batch_size]
    
    @property
    def is_ready(self) -> bool:
        """检查是否已经捕获了graph"""
        return self._ready
