import torch
import torch.nn as nn
try:
    import triton
    import triton.language as tl
except ImportError:
    print("please install triton")


@triton.jit
def fused_gate_up_silu_kernel(
    # Pointers
    input_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    # Dimensions
    M, N, K,  # M: batch_size * seq_len, N: hidden_dim, K: intermediate_dim
    # Strides
    stride_m, stride_n, stride_k,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """
    📌 **Fused gate_proj + up_proj + SiLU** (Triton)

    🔍 **计算**:
        - gate = input @ gate_weight.T
        - up = input @ up_weight.T
        - output = gate * sigmoid(gate) * up

    🧠 **优化**:
        - 合并 gate_proj 和 up_proj 的 GEMM
        - 直接计算 SiLU，避免中间结果写回
        - 使用 Tensor Core (bf16)
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Input mask
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Grouped range for better L2 cache utilization
    GROUP_SIZE_M = BLOCK_M * GROUP_M
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(GROUP_SIZE_M, (pid_m + GROUP_SIZE_M) // GROUP_SIZE_M * GROUP_SIZE_M - pid_m)
    pid_m = first_pid_m + (pid_m - first_pid_m) % group_size_m

    # Accumulators (use fp32 for accumulation)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load input
        input_offs = (offs_m[:, None] * stride_m + offs_k[None, :] * stride_k)
        input_mask = mask_m[:, None] & mask_k[None, :]
        input_val = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

        # Load gate_weight
        gate_weight_offs = (offs_k[:, None] * stride_n + offs_n[None, :] * stride_k)
        gate_weight_mask = mask_k[:, None] & mask_n[None, :]
        gate_weight_val = tl.load(gate_weight_ptr + gate_weight_offs, mask=gate_weight_mask, other=0.0)

        # Load up_weight
        up_weight_offs = (offs_k[:, None] * stride_n + offs_n[None, :] * stride_k)
        up_weight_mask = mask_k[:, None] & mask_n[None, :]
        up_weight_val = tl.load(up_weight_ptr + up_weight_offs, mask=up_weight_mask, other=0.0)

        # GEMM (use Tensor Core if input is bf16)
        acc_gate += tl.dot(input_val, gate_weight_val, out_dtype=tl.float32)
        acc_up += tl.dot(input_val, up_weight_val, out_dtype=tl.float32)

    # SiLU: gate * sigmoid(gate)
    acc_silu = acc_gate * tl.sigmoid(acc_gate.to(tl.float32))
    # Element-wise multiplication: silu * up
    acc_out = acc_silu * acc_up

    # Store output
    output_offs = (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    tl.store(output_ptr + output_offs, acc_out.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


class FusedMLP(nn.Module):
    """
    📌 **Fused MLP** (Qwen-style: gate_proj + up_proj + SiLU + down_proj)

    🔍 **结构**:
        - gate_proj: [B, S, D] -> [B, S, 4D]
        - up_proj: [B, S, D] -> [B, S, 4D]
        - SiLU: gate * sigmoid(gate)
        - down_proj: [B, S, 4D] -> [B, S, D]

    🧠 **优化**:
        - gate_proj + up_proj + SiLU 融合为单个 Triton 内核
        - down_proj 使用 PyTorch (可替换为 CUTLASS)
        - 支持 bf16 和 fp16
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype

        # 权重 (注册为 buffer，避免被优化器更新)
        self.register_buffer(
            "gate_weight",
            torch.randn((hidden_size, intermediate_size), dtype=dtype),
        )
        self.register_buffer(
            "up_weight",
            torch.randn((hidden_size, intermediate_size), dtype=dtype),
        )
        self.register_buffer(
            "down_weight",
            torch.randn((intermediate_size, hidden_size), dtype=dtype),
        )

        # 内核配置 (根据 A100 优化)
        self.block_m = 128
        self.block_n = 128
        self.block_k = 32
        self.group_m = 8
        self.split_k = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        📌 **输入**: x [B, S, D] (bf16/fp16)
        📌 **输出**: output [B, S, D] (bf16/fp16)
        """
        B, S, D = x.shape
        M = B * S
        N = self.intermediate_size
        K = self.hidden_size

        # 输入必须是 contiguous 和 bf16/fp16
        x = x.contiguous().to(self.dtype)
        output = torch.empty((B, S, N), dtype=self.dtype, device=x.device)

        # 启动内核
        grid = (triton.cdiv(M, self.block_m), triton.cdiv(N, self.block_n))
        fused_gate_up_silu_kernel[grid](
            x, self.gate_weight, self.up_weight, output,
            M, N, K,
            x.stride(0), x.stride(1), self.gate_weight.stride(0),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            GROUP_M=self.group_m,
            SPLIT_K=self.split_k,
        )

        # down_proj (使用 PyTorch 的 GEMM，可替换为 CUTLASS)
        output = torch.matmul(output, self.down_weight)  # [B, S, 4D] @ [4D, D] -> [B, S, D]
        return output


# ✅ **测试**
if __name__ == "__main__":
    # 测试参数
    B, S, D, N = 2, 1024, 4096, 11008  # Qwen-7B 的 hidden_size/intermediate_size
    device = "cuda:0"

    # 创建模型
    fused_mlp = FusedMLP(D, N, dtype=torch.bfloat16).to(device)
    mlp = nn.Sequential(
        nn.Linear(D, N, bias=False, dtype=torch.bfloat16),  # gate_proj
        nn.Linear(D, N, bias=False, dtype=torch.bfloat16),  # up_proj
        nn.SiLU(),
        nn.Linear(N, D, bias=False, dtype=torch.bfloat16),  # down_proj
    ).to(device)

    # 复制权重
    fused_mlp.gate_weight.copy_(mlp[0].weight.T)
    fused_mlp.up_weight.copy_(mlp[1].weight.T)
    fused_mlp.down_weight.copy_(mlp[3].weight)

    # 测试输入
    x = torch.randn((B, S, D), dtype=torch.bfloat16, device=device)

    # 预热
    for _ in range(5):
        with torch.no_grad():
            y1 = fused_mlp(x)
            y2 = mlp[3](mlp[2](mlp[0](x) * torch.sigmoid(mlp[0](x)) * mlp[1](x)))

    # 性能比较
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y1 = fused_mlp(x)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / 100 * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y2 = mlp[3](mlp[2](mlp[0](x) * torch.sigmoid(mlp[0](x)) * mlp[1](x)))
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100 * 1000

    # 数值误差
    y2 = mlp[3](mlp[2](mlp[0](x) * torch.sigmoid(mlp[0](x)) * mlp[1](x)))
    error = torch.abs(y1 - y2).mean().item()

    print(f"Fused MLP Time: {fused_time:.4f} ms")
    print(f"Torch MLP Time: {torch_time:.4f} ms")
    print(f"Speedup: {torch_time / fused_time:.2f}x")
    print(f"Mean Error: {error:.6f}")