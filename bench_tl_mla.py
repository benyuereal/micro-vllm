#!/usr/bin/env python3
"""对比 TileLang paged MLA kernel vs 当前 flash_attn_varlen,在 V2-Lite 真实维度下。

目的:确认 TileLang MLA 在 L20 + bs=1/seq=1024(基准场景)是否比当前 flash 路径快,
以及它在更大 batch 下的扩展性。这是 attention 全融合的基础——先确认 flash 这一步用 TileLang 值不值。
"""
import sys, math, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
sys.path.insert(0, "/models/tilelang/examples/deepseek_mla")
import tilelang
from example_mla_decode_paged import mla_decode_tilelang
from flash_attn import flash_attn_varlen_func
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext


def bench(fn, n_iter=300):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000  # us


def bench_graph(fn, n_iter=300):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(20): g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


def main():
    # V2-Lite 维度
    H = 16          # num_heads
    h_kv = 1
    dv = 128        # v_head = qk_nope
    dpe = 64        # qk_rope
    d = dv + dpe    # 192
    block_size = 64  # TileLang kernel 要求 block_size >= block_N 且整除;我们 cache 是 256,这里测 kernel 自带示例
    BLOCK_N = 64
    BLOCK_H = min(64, H)
    num_split = 1
    softmax_scale = d ** -0.5
    dtype = torch.float16  # kernel 写死 fp16

    device = "cuda"
    print(f"V2-Lite MLA: H={H} dv={dv} dpe={dpe} d={d} dtype={dtype}\n", flush=True)

    for (b, seq) in [(1, 1024), (1, 256), (2, 1024), (4, 1024), (8, 1024), (16, 1024)]:
        max_seqlen = seq
        max_seqlen_pad = math.ceil(max_seqlen / 256) * 256
        cache_seqlens = torch.tensor([seq for _ in range(b)], dtype=torch.int32, device=device)
        q = torch.randn(b, H, d, dtype=dtype, device=device)
        block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=dtype, device=device)

        q_nope = q[..., :dv].contiguous()
        q_pe = q[..., dv:].contiguous()
        blocked_k_nope = blocked_k[..., :dv].contiguous()
        blocked_k_pe = blocked_k[..., dv:].contiguous()

        out_partial = torch.empty(b, H, num_split, dv, dtype=dtype, device=device)
        glse = torch.empty(b, H, num_split, dtype=dtype, device=device)
        kernel = mla_decode_tilelang(b, H, h_kv, max_seqlen_pad, dv, dpe, BLOCK_N, BLOCK_H, num_split, block_size, softmax_scale)

        def tl_call():
            return kernel(q_nope.view(-1, H, dv), q_pe.view(-1, H, dpe),
                          blocked_k_nope.view(-1, h_kv, dv), blocked_k_pe.view(-1, h_kv, dpe),
                          block_table, cache_seqlens, glse, out_partial)

        # warmup + 正确性
        try:
            out_tl = tl_call()
        except Exception as ex:
            print(f"bs={b} seq={seq}: TileLang 调用失败 {ex}")
            continue

        # 对比:当前 flash_attn_varlen 路径
        # MLA: q=[b,H,d], k=[b,seq,H,d](kv_head 复制到 H), v=[b,seq,H,dv]
        k_full = blocked_k.view(b, max_seqlen_pad, h_kv, d).expand(-1, -1, H, -1).contiguous()
        v_full = k_full[..., :dv].contiguous()
        cu_q = torch.arange(0, b + 1, dtype=torch.int32, device=device)
        cu_k = torch.zeros(b + 1, dtype=torch.int32, device=device)
        cu_k[1:] = torch.cumsum(cache_seqlens.to(torch.int32), dim=0)
        q_fa = q.view(b, H, d)

        def fa_call():
            return flash_attn_varlen_func(q_fa, k_full.view(b * max_seqlen_pad, H, d),
                                          v_full.view(b * max_seqlen_pad, H, dv),
                                          cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                          max_seqlen_q=1, max_seqlen_k=max_seqlen_pad,
                                          softmax_scale=softmax_scale, causal=False)

        try:
            fa_call()
            fa_ok = True
        except Exception as ex:
            fa_ok = False
            print(f"  (flash 对比跳过: {type(ex).__name__})")

        t_tl = bench(tl_call)
        t_tl_g = bench_graph(tl_call)
        if fa_ok:
            t_fa = bench(fa_call)
            t_fa_g = bench_graph(fa_call)
            print(f"bs={b:2d} seq={seq:4d}: TileLang {t_tl:6.1f}us (graph {t_tl_g:6.1f}) | flash_varlen {t_fa:6.1f}us (graph {t_fa_g:6.1f})")
        else:
            print(f"bs={b:2d} seq={seq:4d}: TileLang {t_tl:6.1f}us (graph {t_tl_g:6.1f})")


if __name__ == "__main__":
    main()
