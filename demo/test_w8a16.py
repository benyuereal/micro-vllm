"""W8A16 int8 GEMV 正确性 + 精度测试。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from kernel.quant import quantize_per_channel, dequantize_per_channel
from kernel.gemv_int8 import w8_linear, gemv_int8_available


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print("gemv_int8_available:", gemv_int8_available())
    for (N, K) in [(1024, 1024), (6144, 1024), (2048, 1024), (3584, 1024), (248320, 1024)]:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        w_int8, scale = quantize_per_channel(w)
        # 参考：bf16 matmul
        ref = (x @ w.t()).float()
        # W8A16：int8 GEMV
        out = w8_linear(x, w_int8, scale)
        # 量化误差参考：用反量化权重做 bf16 matmul
        w_dq = dequantize_per_channel(w_int8, scale)
        ref_dq = (x @ w_dq.t()).float()
        d_kernel = (out.float() - ref_dq).abs().max().item()   # kernel vs 反量化（应≈0，验证 kernel 正确）
        d_quant = (ref_dq - ref).abs().max().item()            # 量化误差
        rel = (ref_dq - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        print(f"N={N:6d} K={K}: kernel_vs_dq={d_kernel:.6f}  quant_err={d_quant:.5f} rel={rel:.5f}")
    # 多 M（prefill 路径，反量化 matmul）
    N, K = 1024, 1024
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, K, device=dev, dtype=torch.bfloat16)
    w_int8, scale = quantize_per_channel(w)
    ref = (x @ w.t()).float()
    out = w8_linear(x, w_int8, scale)
    print(f"M=8 prefill: max_diff={(out.float()-ref).abs().max().item():.5f}")


if __name__ == "__main__":
    main()
