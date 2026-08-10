#!/usr/bin/env python3
"""量化单次 grouped GEMV 的纯 kernel 时间 vs 16次 loop 总时间。

回答：1055us 的 gemv_loop 里，多少是 16 次 launch 开销、多少是纯算、多少是 act 落 HBM？
- 单次 grouped_gate_up + grouped_down 的 CUDA event 纯 kernel 时间（N=1 token, K=6 expert）
- 16 次（N=8 token）串行的总时间
- 对比：如果单次纯算只有 ~3us，16次=48us，那 1055-48≈1000us 全是 launch+HBM 开销 → TileLang persistent 收益巨大
"""
import sys
import torch
sys.path.insert(0, "/models/micro-vllm")
from kernel.grouped_gemv import grouped_gate_up, grouped_down
import torch.nn.functional as F


def bench(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    H, OUT, INTER, E, K = 2048, 2 * 1408, 1408, 64, 6
    dev = "cuda"
    dt = torch.float16

    x1 = torch.randn(1, H, device=dev, dtype=dt)
    e_gu = torch.randn(E, OUT, H, device=dev, dtype=dt) * 0.02
    e_d = torch.randn(E, H, INTER, device=dev, dtype=dt) * 0.02
    idx = torch.randint(0, E, (K,), device=dev, dtype=torch.int64)
    w = torch.rand(K, device=dev, dtype=dt)
    w_ones = torch.ones(K, device=dev, dtype=dt)

    # 单 token 完整 SwiGLU: gate_up + silu*up*w + down
    def single_token():
        gu = grouped_gate_up(x1, e_gu, idx)          # [K, 2*inter]
        gate, up = gu.chunk(2, dim=-1)
        act = F.silu(gate) * up * w.unsqueeze(-1).to(gu.dtype)
        return grouped_down(act, e_d, idx, w_ones)   # [1, H]

    t_single = bench(single_token)
    print(f"单 token 完整 SwiGLU (gate_up+down): {t_single:.2f} us")

    # 单独 gate_up
    def only_gate_up():
        return grouped_gate_up(x1, e_gu, idx)
    t_gu = bench(only_gate_up)
    print(f"  grouped_gate_up only: {t_gu:.2f} us")

    # 单独 down（需要先算 act）
    act_pre = F.silu(grouped_gate_up(x1, e_gu, idx).chunk(2, dim=-1)[0]) * grouped_gate_up(x1, e_gu, idx).chunk(2, dim=-1)[1]
    def only_down():
        return grouped_down(act_pre, e_d, idx, w_ones)
    t_d = bench(only_down)
    print(f"  grouped_down only: {t_d:.2f} us")

    # N=8 token 串行（模拟 decode loop）
    N = 8
    x8 = torch.randn(N, H, device=dev, dtype=dt)
    idx8 = torch.randint(0, E, (N, K), device=dev, dtype=torch.int64)
    w8 = torch.rand(N, K, device=dev, dtype=dt)
    out8 = torch.empty(N, H, device=dev, dtype=dt)

    def loop8():
        for i in range(N):
            ii = idx8[i]
            wi = w8[i]
            gu = grouped_gate_up(x8[i:i+1], e_gu, ii)
            g, u = gu.chunk(2, dim=-1)
            act = F.silu(g) * u * wi.unsqueeze(-1).to(gu.dtype)
            out8[i:i+1] = grouped_down(act, e_d, ii, w_ones)

    t_loop8 = bench(loop8)
    print(f"\nN=8 token 串行 loop: {t_loop8:.2f} us (16 次 kernel)")
    print(f"  单 token × 8 = {t_single*8:.2f} us")
    print(f"  loop 开销: {t_loop8 - t_single*8:.2f} us (launch + elementwise + HBM)")

    # 对比：纯算下限（一次大 GEMM，不分组）
    # gate_up: [8, H] @ [H, OUT] = 理论最小
    W_gu_flat = e_gu[0].T.contiguous()  # [H, OUT] 单 expert
    def flat_gate_up():
        return x8 @ W_gu_flat
    t_flat = bench(flat_gate_up)
    print(f"\n参考: [8,H]@[H,OUT] 单个大 GEMM: {t_flat:.2f} us (tensor core, 无分组开销)")

    print(f"\n=== 结论 ===")
    print(f"单 token SwiGLU 纯 kernel: {t_single:.1f} us")
    print(f"N=8 loop 总: {t_loop8:.1f} us")
    print(f"  若 TileLang persistent 把 16 launch→1, 省掉 loop 开销 {t_loop8 - t_single*8:.0f}us")
    print(f"  理论可降到 ~{t_single*8:.0f}us (纯算) 甚至更低(act 留 smem)")


if __name__ == "__main__":
    main()
