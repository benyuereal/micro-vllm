"""任务 B：verify flash attn 开 split 的 ROI 评估。

flash_attn 2.8.3 的 flash_attn_varlen_func 签名【无 num_splits】（FA3 才有），
所以 2.x 下无法开 split。本脚本量化 verify 里 flash attn 段耗时 vs kv_len，
判断即便能开 split 收益是否 <5%（短上下文下预期无 ROI）。

verify 形状：M=8 query（1 anchor + 7 draft）attend 全上下文 kv_len。
full attention：24 q heads / 4 kv heads / head_dim 256，16 层（interval=4）。
用 paged KV（block_table）对齐真实 _prefill_full 路径。

用法：CUDA_VISIBLE_DEVICES=3 python3 demo/profile_flash_split.py
"""
import os, sys, time
import torch
from flash_attn import flash_attn_varlen_func

NH, KVH, HD = 24, 4, 256
N_LAYERS = 16
M = 8  # verify query 数
BLOCK = 256  # KV block size（对齐 cache_manager DEFAULT_BLOCK_SIZE=256，flash 要求 %256==0）
N_ITER = 20


def bench(kv_len, device):
    # paged KV cache：[n_blocks, block, kvh, hd]
    n_blocks = (kv_len + BLOCK - 1) // BLOCK
    k_cache = torch.randn(n_blocks, BLOCK, KVH, HD, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(n_blocks, BLOCK, KVH, HD, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(n_blocks, dtype=torch.int32, device=device).view(1, -1)
    # 8 query（varlen：cu_seqlens_q=[0,8]），kv 全上下文
    q = torch.randn(M, NH, HD, dtype=torch.bfloat16, device=device)
    cu_q = torch.tensor([0, M], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    # warm
    for _ in range(3):
        flash_attn_varlen_func(q, k_cache, v_cache, cu_q, cu_k, M, kv_len,
                               softmax_scale=HD ** -0.5, causal=True, block_table=block_table)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        flash_attn_varlen_func(q, k_cache, v_cache, cu_q, cu_k, M, kv_len,
                               softmax_scale=HD ** -0.5, causal=True, block_table=block_table)
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - t0) / N_ITER * 1000  # ms / 层
    return per_call


@torch.inference_mode()
def main():
    device = "cuda"
    print("flash_attn 2.8.3 flash_attn_varlen_func 无 num_splits 参数（FA3 才有）→ 2.x 无法开 split")
    print("\n=== verify flash attn 单层耗时（ms/层，%d query attend kv_len）===" % M)
    print("  kv_len   单层ms    16层ms    占 verify(500ms)")
    verify_ms = 500.0
    for kv_len in [125, 512, 2048]:
        per = bench(kv_len, device)
        total16 = per * N_LAYERS
        print(f"  {kv_len:6d}  {per:8.4f}  {total16:8.3f}   {total16/verify_ms*100:6.3f}%")


if __name__ == "__main__":
    main()
