"""
===================================================================
ModelLayerAdapter - vLLM 多模型架构适配器 (极简设计)
===================================================================
"""

import logging
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
from kernel.rmsnorm import rms_norm

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)


class ModelLayerAdapter:
    MODEL_CONFIGS = {
        "qwen": {
            "norm": "ln_1", "attn": "c_attn", "proj": "c_proj", "mlp_norm": "ln_2",
            "qkv_split": True, "qkv_proj": False,
            "mlp": "mlp", "residual": True,
        },
        "qwen2": {
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
        },
        "qwen3": {
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
            "moe": True,
        },
    }

    def __init__(self, model, model_config, device: str, num_heads: int, 
                 head_size: int, kv_num_heads: int, max_batch_size: int = 16):
        self.model = model
        self.config = model_config
        self.device = device
        self.model_type = model_config.model_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self.max_batch_size = max_batch_size
        
        # 关键维度
        self.hidden_dim = model_config.hidden_size
        self.intermediate_size = getattr(model_config, 'intermediate_size', 
                                         self.hidden_dim * 4)  # 默认 4x hidden
        
        # 初始化 attention
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device,
            max_batch_size=max_batch_size
        )
        
        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.cfg = self.MODEL_CONFIGS[self.model_type]
        
        # 预缓存权重
        self._ready = False
        self.prepare(model)
        
        # 初始化缓冲区
        self._init_buffers()
        
        # CUDA Graphs
        self._graphs = {}  # (idx, batch_size) -> graph
        self._graphs_ready = False

    def _init_buffers(self):
        """初始化静态缓冲区"""
        max_b = self.max_batch_size
        
        # 输入输出
        self._hidden = torch.empty((max_b, self.hidden_dim), 
                                   dtype=torch.bfloat16, device=self.device)
        self._residual = torch.empty_like(self._hidden)
        self._output = torch.empty_like(self._hidden)
        
        # Attention
        self._normed_1 = torch.empty_like(self._hidden)
        self._qkv = torch.empty((max_b, 3 * self.hidden_dim), 
                                dtype=torch.bfloat16, device=self.device)
        self._attn_out = torch.empty((max_b, self.hidden_dim), 
                                     dtype=torch.bfloat16, device=self.device)
        
        # MLP
        self._normed_2 = torch.empty_like(self._hidden)
        self._gate_up = torch.empty((max_b, 2 * self.intermediate_size), 
                                    dtype=torch.bfloat16, device=self.device)
        self._mlp_out = torch.empty_like(self._hidden)

    def prepare(self, model):
        """预缓存转置权重"""
        if self._ready:
            return
        
        for idx, block in enumerate(model.transformer.h):
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
        
        logger.info("✅ 预缓存转置权重完成")
        self._ready = True

    def capture_graphs(self, cache_manager, num_layers, batch_sizes=[1, 2, 4, 8, 16]):
        """捕获每个 transformer block 的 CUDA Graph"""
        if self._graphs_ready:
            return
        
        logger.info(f"Capturing {num_layers} blocks x {len(batch_sizes)} batch sizes")
        
        for idx in range(num_layers):
            block = self.model.transformer.h[idx]
            for bs in batch_sizes:
                self._capture_single(block, idx, bs, cache_manager)
        
        self._graphs_ready = True
        logger.info("CUDA Graphs capture completed")

    def _capture_single(self, block, idx, batch_size, cache_manager):
        """捕获单个 block 的完整计算图"""
        g = torch.cuda.CUDAGraph()
        
        # 预热（关键！）
        self._warmup(block, idx, batch_size, cache_manager)
        
        # 获取权重
        w_qkv = block.attn._qkv_w
        b_qkv = block.attn._qkv_b
        w_o = block.attn._o
        w_gu = block.mlp._gu
        w_d = block.mlp._d
        
        # KV cache
        k_cache, v_cache = cache_manager.get(idx)
        
        with torch.cuda.graph(g):
            h = self._hidden[:batch_size]
            
            # === Attention ===
            normed = rms_norm(h, block.ln_1.weight, block.ln_1.eps)
            self._normed_1[:batch_size] = normed
            
            # QKV Linear
            qkv = torch.matmul(self._normed_1[:batch_size], w_qkv)
            if b_qkv is not None:
                qkv = qkv + b_qkv
            self._qkv[:batch_size] = qkv
            
            # Split QKV
            q = qkv[:, :self.hidden_dim].view(batch_size, self.num_heads, self.head_size)
            k = qkv[:, self.hidden_dim:2*self.hidden_dim].view(batch_size, self.kv_num_heads, self.head_size)
            v = qkv[:, 2*self.hidden_dim:].view(batch_size, self.kv_num_heads, self.head_size)
            
            # FlashAttention
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
            ).squeeze(1)
            
            # O Proj
            attn_flat = attn.view(batch_size, -1)
            out = torch.matmul(attn_flat, w_o)
            
            # First Residual
            h = out + self._hidden[:batch_size]
            self._residual[:batch_size] = h
            
            # === MLP ===
            normed = rms_norm(h, block.ln_2.weight, block.ln_2.eps)
            self._normed_2[:batch_size] = normed
            
            # Gate + Up
            gate_up = torch.matmul(self._normed_2[:batch_size], w_gu)
            
            # SwiGLU
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            
            # Down
            mlp_out = torch.matmul(activated, w_d)
            
            # Second Residual
            out = mlp_out + self._residual[:batch_size]
            self._output[:batch_size] = out
        
        self._graphs[(idx, batch_size)] = g

    def _warmup(self, block, idx, batch_size, cache_manager, num_warmup=3):
        """完整预热整个 block"""
        dummy_hidden = torch.randn(batch_size, self.hidden_dim,
                                dtype=torch.bfloat16, device=self.device)
        
        # 获取所有需要的东西
        w_qkv = block.attn._qkv_w
        b_qkv = block.attn._qkv_b
        w_o = block.attn._o
        w_gu = block.mlp._gu
        w_d = block.mlp._d
        k_cache, v_cache = cache_manager.get(idx)
        
        for _ in range(num_warmup):
            h = dummy_hidden
            
            # === 完整 Attention ===
            normed = rms_norm(h, block.ln_1.weight, block.ln_1.eps)
            qkv = torch.matmul(normed, w_qkv)
            if b_qkv is not None:
                qkv = qkv + b_qkv
            
            q = qkv[:, :self.hidden_dim].view(batch_size, self.num_heads, self.head_size)
            k = qkv[:, self.hidden_dim:2*self.hidden_dim].view(batch_size, self.kv_num_heads, self.head_size)
            v = qkv[:, 2*self.hidden_dim:].view(batch_size, self.kv_num_heads, self.head_size)
            
            attn = flash_attn_with_kvcache(
                q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
                k=k.unsqueeze(1), v=v.unsqueeze(1),
                rotary_cos=self.attention._cos_pool,
                rotary_sin=self.attention._sin_pool,
                cache_seqlens=torch.ones(batch_size, dtype=torch.int32, device=self.device),
                block_table=torch.zeros(batch_size, cache_manager.max_seq_blocks, 
                                    dtype=torch.int32, device=self.device),
                causal=True,
            ).squeeze(1)
            
            attn_flat = attn.view(batch_size, -1)
            out = torch.matmul(attn_flat, w_o)
            h = out + h  # residual
            
            # === 完整 MLP ===
            normed = rms_norm(h, block.ln_2.weight, block.ln_2.eps)
            gate_up = torch.matmul(normed, w_gu)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            mlp_out = torch.matmul(activated, w_d)
            out = mlp_out + h  # residual
        
        torch.cuda.synchronize()

    def forward(self, idx, batch_size, hidden_states, cache_manager):
        """使用 CUDA Graph 执行单个 block"""
        # 拷贝输入
        self._hidden[:batch_size].copy_(hidden_states)
        
        # Replay
        key = (idx, batch_size)
        if key not in self._graphs:
            raise RuntimeError(f"Graph not found: idx={idx}, batch={batch_size}")
        
        self._graphs[key].replay()
        
        # 返回输出
        return self._output[:batch_size].clone()

    def process_layer(self, layer, hidden_states, cache_manager, seq_ids,
                  context_lens, token_positions=None, layer_idx=0,
                  current_positions=None):
        """兼容旧接口，使用 CUDA Graph"""
        batch_size = hidden_states.shape[0]
        
        if self._graphs_ready and (layer_idx, batch_size) in self._graphs:
            # 更新 cache_manager 的 buffer
            cache_manager.get_buffer_data(seq_ids, context_lens)
            # 调用 forward，使用关键字参数更清晰
            return self.forward(
                idx=layer_idx,
                batch_size=batch_size,
                hidden_states=hidden_states,
                cache_manager=cache_manager
            ), None
        else:
            # Fallback：抛出错误或实现 eager
            raise RuntimeError(f"Graph not ready for idx={layer_idx}, batch={batch_size}")

 



  