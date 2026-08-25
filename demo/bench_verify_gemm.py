"""隔离 benchmark：verify（M=8）int8 GEMM 各路径耗时。

对比（同一 int8 权重，M=8）：
  1. TileLang int8 GEMM（kernel/gemm_int8.py，verify 默认后端）
  2. Triton int8 GEMM（kernel/gemm_int8_triton.py，verify 备选后端）
  3. 8× M=1 GEMV（bit-exact 参考，权重读 8 次）
  4. dequant→bf16 + cuBLAS GEMM（tensor core，非 bit-exact，速度上限参考）
"""
import os, sys, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from kernel.gemv_int8 import w8_linear, gemv_int8_available
from kernel.gemm_int8_triton import int8_gemm_triton


def bench(fn, iters=10):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    device = "cuda"
    dtype = torch.bfloat16
    M = 8
    print(f"gemv_int8_available={gemv_int8_available()}")
    have_tl = False
    try:
        from kernel.gemm_int8 import int8_gemm
        have_tl = True
    except Exception as e:
        print(f"TileLang import fail: {e}")
    # 代表性层（Qwen3.8-27B）：mlp_gu N=34816 K=5120；gdn_qz N=16384 K=5120；
    # full_qkv N=13312 K=5120；mlp_down N=5120 K=17408
    for name, N, K in [("mlp_gu", 34816, 5120), ("gdn_qz", 16384, 5120),
                       ("full_qkv", 13312, 5120), ("mlp_down", 5120, 17408)]:
        x = torch.randn(M, K, dtype=dtype, device=device)
        w_int8 = (torch.randn(N, K, device=device) * 10).to(torch.int8)
        scale = torch.rand(N, K // 128, dtype=torch.float32, device=device) + 0.5
        out = torch.empty(M, N, dtype=dtype, device=device)

        r = {}
        if have_tl:
            r["tilelang"] = bench(lambda: int8_gemm(x, w_int8, scale, out))
        r["triton"] = bench(lambda: int8_gemm_triton(x, w_int8, scale, out))

        # 8× M=1 GEMV
        def eight_gemv():
            for m in range(M):
                w8_linear(x[m:m+1], w_int8, scale, out[m:m+1], "MICRO_GEMV")
        r["8xGEMV"] = bench(eight_gemv)

        # dequant→bf16 + cuBLAS（非 bit-exact，速度上限）
        sc = scale.repeat_interleave(128, dim=1)
        w_bf16 = (w_int8.float() * sc).to(dtype)
        r["cuBLAS"] = bench(lambda: torch.matmul(x, w_bf16.t(), out=out))

        # 内存下限：int8 权重读 1 次
        w_bytes = N * K
        line = f"{name} (N={N} K={K} W={w_bytes/1e6:.0f}MB): "
        for k in ["tilelang", "triton", "8xGEMV", "cuBLAS"]:
            if k in r:
                line += f"{k}={r[k]:.3f}ms  "
        line += f"mem_floor(864GB/s)={w_bytes/864e6:.1f}ms"
        print(line)


if __name__ == "__main__":
    main()
