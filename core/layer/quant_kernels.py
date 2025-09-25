# File: quant_kernels.py
import torch
import triton
import triton.language as tl


class QuantKernels:
    """
    INT4 GPTQ 量化内核 - Qwen7B 专用
    """

    @staticmethod
    def fused_quant_qkv_proj(
            hidden_states: torch.Tensor,
            qkv_weight: torch.Tensor,
            qkv_scale: torch.Tensor,
            qkv_zero: torch.Tensor,
            num_heads: int,
            head_dim: int,
            group_size: int = 128
    ) -> tuple:
        """
        融合反量化的 QKV 投影内核 (Qwen7B 专用)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 创建输出张量 [B, H, S, D]
        q = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)
        k = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)
        v = torch.empty((batch_size, num_heads, seq_len, head_dim),
                        device=hidden_states.device, dtype=torch.float16)

        # 设置 Triton 网格
        grid = (batch_size, triton.cdiv(seq_len, 16))

        # 启动量化内核
        QuantKernels._fused_quant_qkv_proj_kernel[grid](
            hidden_states,
            qkv_weight,
            qkv_scale,
            qkv_zero,
            q, k, v,
            batch_size, seq_len, hidden_dim,
            num_heads, head_dim,
            group_size,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=32
        )

        return q, k, v

    @staticmethod
    def fused_quant_out_proj(
            attn_output: torch.Tensor,
            out_weight: torch.Tensor,
            out_scale: torch.Tensor,
            out_zero: torch.Tensor,
            group_size: int = 128
    ) -> torch.Tensor:
        """
        融合反量化的输出投影内核 (Qwen7B 专用)
        """
        batch_size, seq_len, hidden_dim = attn_output.shape

        # 创建输出张量
        output = torch.empty_like(attn_output)

        # 设置 Triton 网格
        grid = (batch_size, triton.cdiv(seq_len, 16))

        # 启动量化内核
        QuantKernels._fused_quant_out_proj_kernel[grid](
            attn_output,
            out_weight,
            out_scale,
            out_zero,
            output,
            batch_size, seq_len, hidden_dim,
            group_size,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=32
        )

        return output

    @staticmethod
    @triton.jit
    def _fused_quant_qkv_proj_kernel(
            hidden_states_ptr, qkv_weight_ptr, qkv_scale_ptr, qkv_zero_ptr,
            q_ptr, k_ptr, v_ptr,
            batch_size, seq_len, hidden_dim, num_heads, head_dim,
            group_size: tl.constexpr,
            BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 64, BLOCK_K: tl.constexpr = 32
    ):
        # 计算 PID 和偏移
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        # 计算输入偏移 [B, S, D]
        input_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim

        # 初始化累加器
        acc_q = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        acc_k = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        # 循环处理 K 维度
        for k in range(0, hidden_dim, BLOCK_K):
            # 加载输入块 [BLOCK_M, BLOCK_K]
            input_block = tl.load(
                hidden_states_ptr + input_offset + k,
                mask=[BLOCK_M, BLOCK_K],
                other=0.0
            )

            # 加载量化权重块 (INT4 打包存储)
            weight_offset = k * 3 * hidden_dim
            quant_weight = tl.load(
                qkv_weight_ptr + weight_offset,
                mask=[BLOCK_K, 3 * hidden_dim],
                other=0
            )

            # 加载量化参数
            group_idx = k // group_size
            scale = tl.load(qkv_scale_ptr + group_idx)
            zero = tl.load(qkv_zero_ptr + group_idx)

            # 反量化权重 (INT4 -> FP32)
            weight_fp32 = (quant_weight.to(tl.float32) - zero) * scale

            # 分割 QKV 权重
            q_weight = weight_fp32[:, :hidden_dim]
            k_weight = weight_fp32[:, hidden_dim:2 * hidden_dim]
            v_weight = weight_fp32[:, 2 * hidden_dim:3 * hidden_dim]

            # 矩阵乘法
            acc_q += tl.dot(input_block, q_weight)
            acc_k += tl.dot(input_block, k_weight)
            acc_v += tl.dot(input_block, v_weight)

        # 存储输出
        q_offset = pid_b * num_heads * seq_len * head_dim + pid_s * BLOCK_M * head_dim
        tl.store(q_ptr + q_offset, acc_q, mask=[BLOCK_M, head_dim])
        tl.store(k_ptr + q_offset, acc_k, mask=[BLOCK_M, head_dim])
        tl.store(v_ptr + q_offset, acc_v, mask=[BLOCK_M, head_dim])

    @staticmethod
    @triton.jit
    def _fused_quant_out_proj_kernel(
            attn_output_ptr, out_weight_ptr, out_scale_ptr, out_zero_ptr,
            out_ptr, batch_size, seq_len, hidden_dim, group_size: tl.constexpr,
            BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 64, BLOCK_K: tl.constexpr = 32
    ):
        # 计算 PID 和偏移
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)

        # 计算输入偏移
        input_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim

        # 初始化累加器
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        # 循环处理 K 维度
        for k in range(0, hidden_dim, BLOCK_K):
            # 加载输入块
            input_block = tl.load(
                attn_output_ptr + input_offset + k,
                mask=[BLOCK_M, BLOCK_K],
                other=0.0
            )

            # 加载量化权重
            weight_offset = k * hidden_dim
            quant_weight = tl.load(
                out_weight_ptr + weight_offset,
                mask=[BLOCK_K, hidden_dim],
                other=0
            )

            # 加载量化参数
            group_idx = k // group_size
            scale = tl.load(out_scale_ptr + group_idx)
            zero = tl.load(out_zero_ptr + group_idx)

            # 反量化权重
            weight_fp32 = (quant_weight.to(tl.float32) - zero) * scale

            # 矩阵乘法
            acc += tl.dot(input_block, weight_fp32)

        # 存储输出
        out_offset = pid_b * seq_len * hidden_dim + pid_s * BLOCK_M * hidden_dim
        tl.store(out_ptr + out_offset, acc, mask=[BLOCK_M, BLOCK_K])