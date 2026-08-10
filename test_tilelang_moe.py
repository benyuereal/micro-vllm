#!/usr/bin/env python3
"""验证 tilelang_moe.moe_routed_decode 对齐 PyTorch 参考。

参考实现 = moe.py decode 路径的 routed experts 部分（不含 gate/topk/shared）：
  对每个 token n, 每个 expert k:
    gu = x[n] @ W_gu[idx[n,k]].T        # [2*inter]
    gate, up = gu[:inter], gu[inter:]
    act = silu(gate) * up * w[n,k]       # [inter]
    out[n] += act @ W_d[idx[n,k]].T      # [hidden]
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/models/micro-vllm")
from kernel.tilelang_moe import moe_routed_decode


def ref_routed(x, e_gu, e_d, idx, w_gate):
    """x[N,H], e_gu[E,2*inter,H], e_d[E,H,inter], idx[N,K] long, w_gate[N,K] -> [N,H]"""
    N, H = x.shape
    K = idx.shape[1]
    inter = e_gu.shape[1] // 2
    out = torch.zeros(N, H, dtype=x.dtype, device=x.device)
    for n in range(N):
        for k in range(K):
            e = idx[n, k].item()
            gu = x[n] @ e_gu[e].T          # [2*inter]
            gate, up = gu[:inter], gu[inter:]
            act = F.silu(gate) * up * w_gate[n, k]
            out[n] += act @ e_d[e].T       # [H]
    return out


def main():
    N, H, INTER, E, K = 8, 2048, 1408, 64, 6
    dev = "cuda"
    dt = torch.float16
    torch.manual_seed(42)

    x = torch.randn(N, H, device=dev, dtype=dt) * 0.5
    e_gu = torch.randn(E, 2 * INTER, H, device=dev, dtype=dt) * 0.02
    e_d = torch.randn(E, H, INTER, device=dev, dtype=dt) * 0.02
    idx = torch.randint(0, E, (N, K), device=dev, dtype=torch.int64)
    w_gate = torch.rand(N, K, device=dev, dtype=dt)

    print("compiling TileLang kernel ...")
    out = moe_routed_decode(x, e_gu, e_d, idx, w_gate)
    print("kernel ran, out:", out.shape, out.dtype)

    print("computing reference ...")
    ref_out = ref_routed(x, e_gu, e_d, idx, w_gate)

    diff = (out.float() - ref_out.float()).abs()
    rel = diff / (ref_out.float().abs() + 1e-3)
    print(f"max abs diff: {diff.max().item():.4f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")
    print(f"max rel diff: {rel.max().item():.4f}")
    print(f"ref norm: {ref_out.float().norm().item():.3f}, out norm: {out.float().norm().item():.3f}")

    # 逐 token 检查
    tok_ok = 0
    for n in range(N):
        d = (out[n].float() - ref_out[n].float()).abs().max().item()
        if d < 1.0:
            tok_ok += 1
        else:
            print(f"  token {n}: max diff {d:.4f}")
    print(f"tokens ok (max_diff<1.0): {tok_ok}/{N}")

    # fp16 精度：abs diff < 0.01 且 norm 一致即可（rel diff 在接近 0 的元素上会爆炸）
    ok = diff.max().item() < 0.05 and abs(ref_out.float().norm().item() - out.float().norm().item()) / ref_out.float().norm().item() < 0.01
    print("✅ routed experts 正确" if ok else "❌ 错误")

    if ok:
        from tilelang.profiler import do_bench
        lat = do_bench(lambda: moe_routed_decode(x, e_gu, e_d, idx, w_gate), warmup=30)
        print(f"\nTileLang moe_routed_decode: {lat*1000:.2f} us (N={N}, K={K})")
        print(f"对比: 当前 Triton loop 16 kernel ≈ 1055 us/层 (整层 profile)")


if __name__ == "__main__":
    main()
