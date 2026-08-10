#!/usr/bin/env python3
"""MoE grouped GEMV 微基准：量化单 token / 跨 token 的 IO 与计算，验证 tile 级重叠收益上限。

测三件事：
 1. 单 token：grouped_gate_up + grouped_down 纯 kernel 时间（CUDA event），按 N=1/8/32 看串行 scaling。
 2. 单 token kernel 的 occupancy/带宽实测：实际 HBM 利用率（读 113MB/token 用多久）。
 3. 跨 token 双 stream 重叠上限：把 N 个 token 分到 2 stream（IO↔compute 天然重叠），测能否快接近 2x。
"""
import sys, time
sys.path.insert(0, "/models/micro-vllm")
import torch
from kernel.grouped_gemv import grouped_gate_up, grouped_down

H = 2048; inter = 1536; E = 64; K = 6
dev = "cuda"; dt = torch.bfloat16

# 构造真实形状的权重与输入
torch.manual_seed(0)
e_gu = torch.randn(E, 2 * inter, H, dtype=dt, device=dev) * 0.02
e_d = torch.randn(E, H, inter, dtype=dt, device=dev) * 0.02
gate_w = torch.randn(E, H, dtype=dt, device=dev) * 0.02
w_ones = torch.ones(K, dtype=dt, device=dev)

def run_token(x_i, idx_i, w_i):
    gu = grouped_gate_up(x_i, e_gu, idx_i)
    gate, up = gu.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate) * up * w_i.unsqueeze(-1).to(gu.dtype)
    return grouped_down(act, e_d, idx_i, w_ones)

def bench(N, iters=200, warmup=30):
    xs = [torch.randn(1, H, dtype=dt, device=dev) for _ in range(N)]
    idxs = [torch.randint(0, E, (K,), dtype=torch.int64, device=dev) for _ in range(N)]
    ws = [torch.rand(K, dtype=dt, device=dev) for _ in range(N)]
    # warmup
    for _ in range(warmup):
        for i in range(N):
            run_token(xs[i], idxs[i], ws[i])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        for i in range(N):
            run_token(xs[i], idxs[i], ws[i])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per outer iter

def bench_two_stream(N, iters=200, warmup=30):
    """把 N 个 token 分到 2 stream 交错（偶数 stream0, 奇数 stream1），看是否接近 2x。"""
    xs = [torch.randn(1, H, dtype=dt, device=dev) for _ in range(N)]
    idxs = [torch.randint(0, E, (K,), dtype=torch.int64, device=dev) for _ in range(N)]
    ws = [torch.rand(K, dtype=dt, device=dev) for _ in range(N)]
    s0 = torch.cuda.Stream(); s1 = torch.cuda.Stream()
    for _ in range(warmup):
        with torch.cuda.stream(s0):
            for i in range(0, N, 2): run_token(xs[i], idxs[i], ws[i])
        with torch.cuda.stream(s1):
            for i in range(1, N, 2): run_token(xs[i], idxs[i], ws[i])
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.cuda.stream(s0):
            for i in range(0, N, 2): run_token(xs[i], idxs[i], ws[i])
        with torch.cuda.stream(s1):
            for i in range(1, N, 2): run_token(xs[i], idxs[i], ws[i])
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

for N in [1, 8, 32]:
    t_serial = bench(N)
    t_2s = bench_two_stream(N) if N >= 2 else t_serial
    bytes_per_token = K * (2*inter*H + H*inter) * 2
    total_bytes = N * bytes_per_token
    hbm_util = total_bytes / (t_serial * 1e-3) / 864e9 * 100  # L20 ~864GB/s
    print(f"N={N:>3}  serial={t_serial*1000:7.1f}us  2stream={t_2s*1000:7.1f}us  speedup={t_serial/t_2s:.2f}x  "
          f"HBM_util={hbm_util:.1f}%  ({total_bytes/1e6:.0f}MB/{t_serial*1e3:.0f}us)")
