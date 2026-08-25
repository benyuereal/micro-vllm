"""prefill GEMM 路径对比：dequant+bf16 matmul vs TileLang int8 GEMM（M=61）。
代表 shape：mlp_gu N=34816 K=5120（27B 最大线性）。"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch

device = "cuda"
M, N, K = 61, 34816, 5120
x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
w_int8 = (torch.randn(N, K, device=device) * 10).to(torch.int8)
scale = (torch.rand(N, K // 128, device=device) * 0.1 + 0.01).float()


def timeit(fn, iters=20):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# 1. dequant + bf16 matmul（当前 prefill 路径）
def dequant_matmul():
    sc = scale.repeat_interleave(128, dim=1)
    w = (w_int8.float() * sc).to(torch.bfloat16)
    return torch.matmul(x, w.t())

# 2. 只 dequant（隔离反量化成本）
def dequant_only():
    sc = scale.repeat_interleave(128, dim=1)
    return (w_int8.float() * sc).to(torch.bfloat16)

# 3. 只 matmul（用预 dequant 的 bf16 权重）
sc = scale.repeat_interleave(128, dim=1)
w_bf16 = (w_int8.float() * sc).to(torch.bfloat16)
def matmul_only():
    return torch.matmul(x, w_bf16.t())

# 4. TileLang int8 GEMM
from kernel.gemm_int8 import int8_gemm
def int8_gemm_tl():
    return int8_gemm(x, w_int8, scale)

print(f"shape M={M} N={N} K={K} (mlp_gu)")
print(f"  dequant+matmul (当前 prefill): {timeit(dequant_matmul):.3f} ms")
print(f"    其中 dequant only:           {timeit(dequant_only):.3f} ms")
print(f"    其中 matmul only (bf16):     {timeit(matmul_only):.3f} ms")
print(f"  TileLang int8 GEMM:            {timeit(int8_gemm_tl):.3f} ms")

# 数值对比
ref = dequant_matmul()
got = int8_gemm_tl()
print(f"  maxdiff int8_gemm vs dequant_matmul: {(ref-got).abs().max().item():.5f}")
print(f"  相对误差: {((ref-got).abs().max()/ref.abs().max()).item():.4f}")
