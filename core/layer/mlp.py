"""
===================================================================
MLP Forward - Qwen MLP 编译优化实现
===================================================================

📌 **核心功能**：
   - Attention 输出投影 + MLP 的编译融合
   - 专为 Qwen 模型优化
   - 针对 CUDA Graphs 优化 (防止 tensor overwriting)

⚡ **性能优化**：
   - torch.compile fullgraph 模式
   - 算子融合减少内存访问
   - 静态 shape 优化
"""
import torch
import torch.nn.functional as F
from kernel.rmsnorm import rms_norm


def compiled(hidden_dim: int = 4096):
    """
    📌 **编译 MLP Forward 函数**
    
    🔍 **参数**:
        - hidden_dim: 隐藏层维度 (Qwen: 4096)
    
    ✅ **返回**:
        - 编译后的 MLP forward 函数
    """
    
    def mlp_forward(layer, hidden, attn_res, attn_out):
        """
        MLP 实现 (会被 torch.compile 编译)
        """
        batch_size = hidden.shape[0]
        
        # Attention 输出投影 + 残差连接
        hidden = attn_res + torch.matmul(attn_out.view(batch_size, -1), layer.attn.c_proj.weight.t().contiguous()).unsqueeze(1)
        
        # LayerNorm + MLP
        normed = rms_norm(hidden, layer.ln_2.weight, layer.ln_2.eps)
        x = normed.view(-1, hidden_dim)
        
        # MLP 投影 (Gate + Up 融合)
        gate_up = torch.matmul(x, layer.mlp._gu)
        up, gate = gate_up.chunk(2, dim=-1)
        
        # SwiGLU 激活
        output = torch.matmul(F.silu(gate) * up, layer.mlp._d)
        
        # 关键：Clone output 防止 CUDA graphs buffer overwriting
        result = (hidden + output.view(hidden.shape)).clone()
        return result
    
    # 编译 MLP 函数
    return torch.compile(
        mlp_forward,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )


class MLPForward:
    """
    📌 **MLP Forward 封装类**
    """
    
    def __init__(self, hidden_dim: int = 4096):
        self.hidden_dim = hidden_dim
        self._forward = compiled(hidden_dim)
    
    def __call__(self, layer, hidden, attn_res, attn_out):
        return self._forward(layer, hidden, attn_res, attn_out)
