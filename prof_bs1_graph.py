#!/usr/bin/env python3
"""bs=1 graph 下整层各段真实 GPU 时间。

基准是单请求 bs=1 (72.2 tok/s, 13.47ms/step)。量 graph 下 attention/ffn/qkv 各段，
确定 bs=1 真正瓶颈。方法：单独 capture 每段的 CUDA graph，测 replay 时间（graph 内纯 GPU 时间），
剔除 CPU launch/同步开销，得到与基准可比的纯 GPU 段耗时。
"""
import sys
import time
import torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext


def bench_graph(fn_build, n_iter=300):
    """fn_build: graph capture 内调用。返回单次 replay us。"""
    # warmup（不在 graph 内）
    for _ in range(5):
        fn_build()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn_build()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000  # us


def main():
    print("Loading engine ...", flush=True)
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)

    # 跑到 decode 状态（让 cache_manager 有合法 block_table / seqlens）
    engine.add_request("写SpringBoot文件上传代码", max_tokens=150, temperature=0.0)
    for _ in range(300):
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b:
            time.sleep(0.001); continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break
    for _ in range(3):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)

    gr = engine.graph_runner
    adapter = gr.adapter
    blocks = adapter.blocks(gr.model)
    H = adapter._hidden
    bs = 1
    cm = engine.cache_manager
    bt_buf = cm._block_table_buffer

    # graph capture 需要 _cur_bucket_maxlen 已设（capture 时已设，但保险）
    gr._cur_bucket_maxlen = gr._deepseek_fixed_maxlen

    # 找第一个 MoE 层和第一个 dense 层
    moe_block = next(b for b in blocks if getattr(b.mlp, "_is_moe", False))
    dense_block = next(b for b in blocks if not getattr(b.mlp, "_is_moe", False))

    # 准备输入（用 gr 的常驻 buffer，与 graph capture 一致）
    gr._h_buf[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    gr._residual[:bs] = torch.randn(bs, H, device="cuda", dtype=engine.dtype)
    h = gr._h_buf[:bs]
    residual = gr._residual[:bs]

    # warmup 各段（建立 attn._q_cache 等临时状态）
    adapter.compute_qkv(dense_block, h, gr, bs)
    attn_out = adapter.attention(gr._h_buf[:bs], dense_block, 0, bs, gr, cm, bt_buf)
    adapter.compute_ffn(dense_block, attn_out, residual, gr, bs, False)
    adapter.compute_qkv(moe_block, h, gr, bs)
    adapter.compute_ffn(moe_block, attn_out, residual, gr, bs, False)
    adapter.compute_next_qkv(moe_block, attn_out, residual, gr, bs)
    torch.cuda.synchronize()

    print(f"\n=== bs=1 graph 下单层各段 GPU 时间 ===\n", flush=True)

    # qkv (dense 层, 首层)
    t_qkv = bench_graph(lambda: adapter.compute_qkv(dense_block, h, gr, bs))
    print(f"qkv  (norm+q_proj+kv_a_proj):            {t_qkv:7.1f} us")

    # attention — 需要先 compute_qkv 设好 _q_cache（graph 外 warmup 一次即可，因为 attention
    # 不改 _q_cache）。但 graph replay 重复执行 attention 会重复 store_latent / 改 cache，
    # 不影响计时（写同位置）。
    adapter.compute_qkv(dense_block, h, gr, bs)
    t_attn = bench_graph(lambda: adapter.attention(gr._h_buf[:bs], dense_block, 0, bs, gr, cm, bt_buf))
    print(f"attention (store+gather+kvb+rope+flash+oproj): {t_attn:7.1f} us")

    # ffn dense
    t_ffn_dense = bench_graph(lambda: adapter.compute_ffn(dense_block, attn_out, residual, gr, bs, False))
    print(f"ffn dense (norm+SwiGLU):                 {t_ffn_dense:7.1f} us")

    # ffn MoE
    adapter.compute_qkv(moe_block, h, gr, bs)
    t_ffn_moe = bench_graph(lambda: adapter.compute_ffn(moe_block, attn_out, residual, gr, bs, False))
    print(f"ffn MoE  (norm+MoE):                     {t_ffn_moe:7.1f} us")

    # next_qkv
    t_next = bench_graph(lambda: adapter.compute_next_qkv(moe_block, attn_out, residual, gr, bs))
    print(f"next_qkv (norm+q_proj+kv_a_proj):        {t_next:7.1f} us")

    print(f"\n--- 推算 27 层整步（3 dense + 24 MoE）---")
    layer_dense = t_qkv + t_attn + t_ffn_dense + t_next
    layer_moe = t_qkv + t_attn + t_ffn_moe + t_next
    total = 3 * layer_dense + 24 * layer_moe
    print(f"  dense 层: {layer_dense:7.1f} us  (qkv {t_qkv:.0f} + attn {t_attn:.0f} + ffn {t_ffn_dense:.0f} + next {t_next:.0f})")
    print(f"  MoE 层:   {layer_moe:7.1f} us  (qkv {t_qkv:.0f} + attn {t_attn:.0f} + ffn {t_ffn_moe:.0f} + next {t_next:.0f})")
    print(f"  3 dense + 24 MoE = {total:.0f} us = {total/1000:.2f} ms")
    print(f"  基准 13.47ms → 推算占 {total/1000/13.47*100:.0f}%")
    print(f"\n  MoE 层内: attention {t_attn/layer_moe*100:.0f}%  ffn {t_ffn_moe/layer_moe*100:.0f}%  qkv+next {(t_qkv+t_next)/layer_moe*100:.0f}%")
    print(f"  → bs=1 瓶颈: attention {t_attn:.0f}us vs ffn(MoE) {t_ffn_moe:.0f}us vs qkv+next {t_qkv+t_next:.0f}us")


if __name__ == "__main__":
    main()
