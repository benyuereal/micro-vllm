"""
===================================================================
QKV Forward - Qwen QKV 编译优化实现
===================================================================

📌 **核心功能**：
   - LayerNorm + QKV 投影的编译融合
   - 专为 Qwen 模型优化

⚡ **性能优化**：
   - torch.compile fullgraph 模式
   - 算子融合减少内存访问
   - 静态 shape 优化
"""
import torch
from kernel.rmsnorm import rms_norm


def compiled(num_heads: int, head_size: int, kv_num_heads: int):
    """
    📌 **编译 QKV Forward 函数**
    
    🔍 **参数**:
        - num_heads: 注意力头数
        - head_size: 每个头的维度
        - kv_num_heads: KV 头数 (GQA 支持)
    
    ✅ **返回**:
        - 编译后的 QKV forward 函数
    """
    
    def qkv_forward(layer, hidden_states):
        """
        QKV 实现 (会被 torch.compile 编译)
        """
        residual = hidden_states
        
        # Qwen: LayerNorm + QKV 融合投影
        hidden_states = rms_norm(hidden_states, layer.ln_1.weight, layer.ln_1.eps)
        qkv = layer.attn.c_attn(hidden_states)
        
        hidden_size = qkv.shape[-1] // 3
        q, k, v = qkv.split(hidden_size, dim=-1)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Reshape: [B, S, D] -> [B, H, S, D]
        q = q.view(batch_size, num_heads, head_size).contiguous()
        k = k.view(batch_size, kv_num_heads, head_size).contiguous()
        v = v.view(batch_size, kv_num_heads, head_size).contiguous()
        
        return hidden_states, residual, q, k, v
    
    # 编译函数
    return torch.compile(
        qkv_forward,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )


class QKVForward:
    """
    📌 **QKV Forward 封装类**
    """
    
    def __init__(self, num_heads: int, head_size: int, kv_num_heads: int):
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads
        self._forward = compiled(num_heads, head_size, kv_num_heads)
    
    def __call__(self, layer, hidden_states):
        """调用编译后的 QKV"""
        return self._forward(layer, hidden_states)
