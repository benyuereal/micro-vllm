import torch


class RoPE:
    """纯粹的 RoPE 旋转类"""

    def __init__(self):
        pass

    def forward(self, q, k, cos, sin, offset: int = 0):
        """应用旋转位置编码。

        参数:
            q, k: 形状 [batch, seq_len, heads, dim]
            cos, sin: 形状 [max_pos, dim//2] 或 [max_pos, dim]（行 i 对应位置 i）
            offset: chunked prefill 时该 chunk 在原 prompt 中的起始位置
                    （已 prefill 的 token 数）。非 chunked 路径 offset=0 行为不变。

        返回:
            q_rot, k_rot: 旋转后的 q 和 k
        """
        seq_len = q.shape[1]

        # 调整 cos/sin 形状用于广播：从 offset 行开始取 seq_len 行
        cos = cos[offset:offset + seq_len][None, :, None, :]
        sin = sin[offset:offset + seq_len][None, :, None, :]
        
        # 标准两半旋转
        if cos.shape[-1] == q.shape[-1] // 2:
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            
            q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        else:
            # 兼容完整维度模式
            q_rot = (q * cos) + (self._rotate_half(q) * sin)
            k_rot = (k * cos) + (self._rotate_half(k) * sin)
            
        return q_rot, k_rot

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)