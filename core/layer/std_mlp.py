import torch
import torch.nn.functional as F

try:
    from kernel.rmsnorm import rms_norm
except ImportError:
    def rms_norm(x, weight, eps=1e-6):
        return F.rms_norm(x, x.shape[-1:], weight, eps)

class MLP:
    """标准 Python MLP（与给定代码完全一致）"""
    @staticmethod
    def forward(hidden, attn_out, attn_proj_weight, norm_weight, gate_up_weight, down_weight, eps=1e-6):
        """
        Args:
            hidden: [B, 1, D] - 输入 hidden_states
            attn_out: [B, H, head_size] - Attention 输出
            attn_proj_weight: [D, D] - attention projection weight (已转置)
            norm_weight: [D] - RMSNorm weight
            gate_up_weight: [D, 2*I] - MLP gate_up weight (已转置)
            down_weight: [I, D] - MLP down weight (已转置)
            eps: float - RMSNorm epsilon
        """
        batch_size = hidden.shape[0]
        
        # 1. Attention Projection + Residual
        hidden_states = attn_res = hidden + torch.matmul(
            attn_out.view(batch_size, -1), 
            attn_proj_weight
        ).unsqueeze(1)
        
        # 2. RMSNorm
        normalized = rms_norm(hidden_states, norm_weight, eps)
        
        # 3. MLP Core
        x = normalized.view(-1, 4096)
        gate_up = torch.matmul(x, gate_up_weight)
        up, gate = gate_up.chunk(2, dim=-1)
        output = torch.matmul(F.silu(gate) * up, down_weight)
        
        # 4. Residual
        return attn_res + output.view(hidden.shape)