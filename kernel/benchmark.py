#!/usr/bin/env python3
"""
Benchmark: TritonDecodeAttention vs flash_attn_with_kvcache
Qwen-7B decode scenario — MHA, head_dim=128, paged KV cache.

Usage:
    python kernel/benchmark.py
"""

import json
import time
from typing import Callable

import torch

from flash import FlashAttention


# ── Data helpers ───────────────────────────────────────────────────────────────

def make_paged_kv_cache(
        batch_size: int, seq_len: int,
        num_kv_heads: int, head_size: int, block_size: int,
        device: str = "cuda",
) -> tuple:
    """Return (k_cache, v_cache, block_tables, context_lens) in paged layout."""
    num_blocks     = (batch_size * seq_len + block_size - 1) // block_size
    max_num_blocks = (seq_len + block_size - 1) // block_size
    k_cache      = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                               device=device, dtype=torch.float16)
    v_cache      = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                               device=device, dtype=torch.float16)
    block_tables = torch.randint(0, num_blocks, (batch_size, max_num_blocks),
                                 device=device, dtype=torch.int32)
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    return k_cache, v_cache, block_tables, context_lens


# ── Timing ─────────────────────────────────────────────────────────────────────

def _measure_ms(fn: Callable, warmup: int = 5, iters: int = 50) -> float:
    """Return mean kernel latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ── Per-kernel benchmarks ──────────────────────────────────────────────────────

def bench_triton(
        batch_size: int, num_heads: int, num_kv_heads: int,
        head_size: int, seq_len: int, block_size: int = 16,
        device: str = "cuda",
) -> dict:
    attn           = FlashAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                    head_size=head_size, block_size=block_size)
    q              = torch.randn(batch_size, num_heads, head_size, device=device, dtype=torch.float16)
    kc, vc, bt, cl = make_paged_kv_cache(batch_size, seq_len, num_kv_heads, head_size, block_size, device)

    latency_ms = _measure_ms(lambda: attn.forward(q, kc, vc, bt, cl))

    # Effective HBM bytes: Q + K + V + O  (fp16 = 2 bytes each)
    bw_bytes = (batch_size * num_heads * head_size * 2                         # Q
                + 2 * batch_size * seq_len * num_kv_heads * head_size * 2     # K + V
                + batch_size * num_heads * head_size * 2)                      # O
    return {
        "batch_size":       batch_size,
        "seq_len":          seq_len,
        "latency_ms":       latency_ms,
        "throughput_tok_s": batch_size / (latency_ms / 1e3),
        "bw_gbs":           bw_bytes / 1e9 / (latency_ms / 1e3),
        "num_splits":       attn.num_splits(seq_len),
    }


def bench_flashattn(
        batch_size: int, num_heads: int, num_kv_heads: int,
        head_size: int, seq_len: int, flash_fn: Callable,
        device: str = "cuda",
) -> float:
    """Return latency (ms) for flash_attn_with_kvcache with dense KV layout."""
    q    = torch.randn(batch_size, 1, num_heads, head_size, device=device, dtype=torch.float16)
    kc   = torch.randn(batch_size, seq_len, num_kv_heads, head_size, device=device, dtype=torch.float16)
    vc   = torch.randn(batch_size, seq_len, num_kv_heads, head_size, device=device, dtype=torch.float16)
    seql = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    return _measure_ms(lambda: flash_fn(q=q, k_cache=kc, v_cache=vc, cache_seqlens=seql, causal=True))


# ── Full sweep ─────────────────────────────────────────────────────────────────

# (batch_size, seq_len) — Qwen-7B decode configs
CONFIGS = [
    (32,    512),
    (32,   1024),
    (32,   2048),
    (32,   4096),
    (32,   8192),
    (32,  16384),
    (32,  32768),
    (16,  65536),
    (8,  131072),
]


def run_benchmark(save_path: str = "benchmark_results.json") -> list:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device: {torch.cuda.get_device_name()}\n")

    try:
        from flash_attn import flash_attn_with_kvcache
        has_ref = True
        print("flash_attn found — benchmarking both\n")
    except ImportError:
        has_ref = False
        print("flash_attn not found — Triton only\n")

    results = []
    for B, S in CONFIGS:
        try:
            r = bench_triton(B, 32, 32, 128, S)
            if has_ref:
                r["ref_ms"] = bench_flashattn(B, 32, 32, 128, S, flash_attn_with_kvcache)
            results.append(r)

            ref_str = (f"  flash={r['ref_ms']:.3f}ms  {r['ref_ms']/r['latency_ms']:.2f}x"
                       if has_ref else "")
            print(f"seq={S:>7}  batch={B}  triton={r['latency_ms']:.3f}ms"
                  f"{ref_str}  bw={r['bw_gbs']:.0f}GB/s  splits={r['num_splits']}")
        except Exception as e:
            print(f"seq={S} batch={B}: ERROR — {e}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    if has_ref:
        print(f"{'Seq':<9} {'Batch':<7} {'Triton(ms)':<13} {'Flash(ms)':<12} {'Speedup':<10} {'BW(GB/s)'}")
        print("-" * 80)
        for r in results:
            ref = r.get("ref_ms", 0.0)
            print(f"{r['seq_len']:<9} {r['batch_size']:<7} {r['latency_ms']:<13.3f} "
                  f"{ref:<12.3f} {ref/r['latency_ms']:.2f}x{'':<6} {r['bw_gbs']:.1f}")
    else:
        print(f"{'Seq':<9} {'Batch':<7} {'Triton(ms)':<13} {'BW(GB/s)'}")
        print("-" * 50)
        for r in results:
            print(f"{r['seq_len']:<9} {r['batch_size']:<7} {r['latency_ms']:<13.3f} {r['bw_gbs']:.1f}")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {save_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
