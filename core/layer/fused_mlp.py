import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    print("please install triton")
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np



# ========================================
# 1. Triton Kernels (SwiGLU + GEMM)
# ========================================

@triton.jit
def swiglu_kernel(
        x_ptr, y_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 手动实现 FP16 兼容的 Sigmoid（避免 tl.sigmoid 的 FP32 限制）
    y_fp32 = y.to(tl.float32)  # 转为 FP32 计算
    sigmoid_y = 1.0 / (1.0 + tl.exp(-y_fp32))
    output = x * (y * sigmoid_y.to(x.dtype))  # 动态匹配输入类型

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        A_DTYPE: tl.constexpr,  # 新增：输入类型
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(A_DTYPE)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# ========================================
# 2. Triton MLP Class (Qwen-7B专用)
# ========================================

class TritonQwenMLP:
    def __init__(self, device='cuda', model=None):
        self.hidden_size = 4096
        self.ffn_hidden_size = 11008
        self.device = device

        # A100优化配置
        self.gemm_block_m = 128
        self.gemm_block_n = 128
        self.gemm_block_k = 64
        self.relu_block_size = 2048

        # 预分配缓冲区（最大支持batch=8, seq=2048）
        self.buffer_gate = torch.empty((8 * 2048, 11008), device=device, dtype=torch.bfloat16)
        self.buffer_up = torch.empty((8 * 2048, 11008), device=device, dtype=torch.bfloat16)
        self.buffer_out = torch.empty((8 * 2048, 4096), device=device, dtype=torch.bfloat16)

        # 加载Qwen-7B权重
        self.load_qwen_weights(model)

    def load_qwen_weights(self, model):
        """适配 Qwen-7B 的权重加载路径和属性名"""

        # 打印第一层的 MLP 结构
        first_mlp = model.transformer.h[0].mlp
        print("🔍 Qwen-7B MLP 结构:", first_mlp)
        print("🔍 MLP 属性:", [attr for attr in dir(first_mlp) if 'weight' in attr])

        # Qwen-7B 的层路径: model.transformer.h
        layers = model.transformer.h
        self.weights = []
        for layer in layers:
            w1 = layer.mlp.w1.weight.data.T.to(self.device)  # 保留 BF16
            w2 = layer.mlp.c_proj.weight.data.T.to(self.device)  # 保留 BF16
            w3 = layer.mlp.w2.weight.data.T.to(self.device)  # 保留 BF16
            self.weights.append((w1, w2, w3))



    def _matmul(self, A, B, out):
        M, K = A.shape
        K, N = B.shape
        for m in range(0, M, 2048):
            for n in range(0, N, 2048):
                m_end = min(m + 2048, M)
                n_end = min(n + 2048, N)
                A_block = A[m:m_end, :]
                B_block = B[:, n:n_end]
                out_block = out[m:m_end, n:n_end]
                grid = lambda meta: (triton.cdiv(m_end - m, meta['BLOCK_SIZE_M']) *
                                     triton.cdiv(n_end - n, meta['BLOCK_SIZE_N']),)
                gemm_kernel[grid](
                    A_block, B_block, out_block,
                    m_end - m, n_end - n, K,
                    A_block.stride(0), A_block.stride(1),
                    B_block.stride(0), B_block.stride(1),
                    out_block.stride(0), out_block.stride(1),
                    BLOCK_SIZE_M=self.gemm_block_m,
                    BLOCK_SIZE_N=self.gemm_block_n,
                    BLOCK_SIZE_K=self.gemm_block_k,
                    A_DTYPE=tl.bfloat16,  # 如果输入是 BF16
                )
        return out

    def forward(self, x, layer_idx):
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.hidden_size)
        w1, w2, w3 = self.weights[layer_idx]

        # 1. gate_proj + up_proj
        gate = self._matmul(x, w1, self.buffer_gate[:x.shape[0], :])
        up = self._matmul(x, w3, self.buffer_up[:x.shape[0], :])

        # 2. SwiGLU
        n_elements = x.shape[0] * self.ffn_hidden_size
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        swiglu_kernel[grid](
            gate, up, gate,
            n_elements, BLOCK_SIZE=self.relu_block_size
        )

        # 3. down_proj
        out = self._matmul(gate, w2, self.buffer_out[:x.shape[0], :])

        return out.view(batch_size, seq_len, self.hidden_size)


# ========================================
# 3. 分步验证代码
# ========================================

def test_kernels():
    print("🔍 验证 SwiGLU 内核...")
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    y = torch.randn(1024, device='cuda', dtype=torch.float16)
    triton_out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    swiglu_kernel[grid](x, y, triton_out, n_elements, BLOCK_SIZE=2048)

    # PyTorch 参考（同样转为 FP32 计算）
    with torch.no_grad():
        y_fp32 = y.float()
        torch_out = x * (y * torch.sigmoid(y_fp32).half())

    assert torch.allclose(triton_out, torch_out, atol=1e-3), "SwiGLU 验证失败"
    print("✅ SwiGLU 内核验证通过")

    # GEMM 验证保持不变...
    print("\n🔍 验证 GEMM 内核...")
    A = torch.randn((256, 128), device='cuda', dtype=torch.float16)
    B = torch.randn((128, 256), device='cuda', dtype=torch.float16)
    triton_C = torch.empty((256, 256), device='cuda', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(256, meta['BLOCK_SIZE_M']) * triton.cdiv(256, meta['BLOCK_SIZE_N']),)
    gemm_kernel[grid](
        A, B, triton_C,
        256, 256, 128,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        triton_C.stride(0), triton_C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,A_DTYPE=tl.bfloat16,  # 如果输入是 BF16
    )
    torch_C = torch.matmul(A, B)
    assert torch.allclose(triton_C, torch_C, atol=1e-2), "GEMM 验证失败"
    print("✅ GEMM 内核验证通过")


def test_end_to_end():
    print("\n🔍 端到端验证 (Qwen-7B MLP)...")

    # 从本地加载 PyTorch 模型
    from transformers import AutoModelForCausalLM, AutoConfig
    model_path = "/root/Qwen-7B-Chat"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )

    triton_mlp = TritonQwenMLP(model=model)
    x = torch.randn(2, 32, 4096, device='cuda', dtype=torch.bfloat16)  # 改为 BF16

    with torch.no_grad():
        # ✅ 全部使用 w1/w2/c_proj
        mlp = model.transformer.h[0].mlp
        gate = mlp.w1(x)  # gate_proj
        up = mlp.w2(x)  # up_proj
        torch_out = mlp.c_proj(gate * torch.sigmoid(up) * up)  # down_proj
        triton_out = triton_mlp.forward(x, layer_idx=0)

    print(f"PyTorch 输出范围: {torch_out.min():.4f} ~ {torch_out.max():.4f}")
    print(f"Triton 输出范围: {triton_out.min():.4f} ~ {triton_out.max():.4f}")
    print(f"最大绝对误差: {torch.abs(torch_out - triton_out).max():.4f}")
    print(f"误差 > 1e-2 的元素占比: {(torch.abs(torch_out - triton_out) > 1e-2).float().mean().item():.2%}")
    assert torch.allclose(torch_out, triton_out, atol=1e-1), "端到端验证失败"
    print("✅ 端到端验证通过")

    del model
    torch.cuda.empty_cache()


# ========================================
# 4. 性能测试 (A100)
# ========================================

def benchmark():
    print("\n🚀 性能测试 (A100 40GB)...")
    # 从本地加载 PyTorch 模型
    from transformers import AutoModelForCausalLM, AutoConfig
    model_path = "/root/Qwen-7B-Chat"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    triton_mlp = TritonQwenMLP(model=model)
    x = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)

    # 预热
    for _ in range(5):
        _ = triton_mlp.forward(x, layer_idx=0)

    # 测量
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = triton_mlp.forward(x, layer_idx=0)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / 100

    print(f"延迟: {latency:.2f}ms | 吞吐量: {1000 / latency * 8 * 2048:.0f} tokens/sec")
    print(f"显存占用: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")


# ========================================
# 5. 运行所有验证
# ========================================

if __name__ == "__main__":
    test_kernels()
    test_end_to_end()
    benchmark()

    # 示例使用
    print("\n📌 示例使用:")
    # 从本地加载 PyTorch 模型
    from transformers import AutoModelForCausalLM, AutoConfig

    model_path = "/root/Qwen-7B-Chat"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )
    mlp = TritonQwenMLP(model=model)
    x = torch.randn(1, 10, 4096, device='cuda', dtype=torch.float16)
    output = mlp.forward(x, layer_idx=0)
    print(f"输入: {x.shape} → 输出: {output.shape}")