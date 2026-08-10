#!/usr/bin/env python3
"""隔离测试 kernel 的 rmsnorm+kvb+RoPE 三步，对比 PyTorch 参考。
构造单 block（block_N=64）latent，跑精简 kernel 输出 k_nope/v/k_pe_rot，逐项比对。"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/models/micro-vllm")
import tilelang
import tilelang.language as T
from core.engine import InferenceEngine

dtype = T.bfloat16
accum = T.float32
block_N = 64; H = 16; kv_lora = 512; qk_rope = 64; qk_nope = 128; v_head = 128
K_TILE = 128; half = qk_rope // 2
kvb_out = qk_nope + v_head


@tilelang.jit(out_idx=[5], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def make_kvb_kernel():
    @T.prim_func
    def main(
        Latent: T.Tensor([block_N, 1, kv_lora + qk_rope], dtype),
        kva_ln_w: T.Tensor([kv_lora], dtype),
        kvb_w: T.Tensor([H * kvb_out, kv_lora], dtype),
        cos_k: T.Tensor([block_N, qk_rope], dtype),
        sin_k: T.Tensor([block_N, qk_rope], dtype),
        Out: T.Tensor([block_N, qk_nope + v_head + qk_rope], dtype),  # 打包 k_nope|v|k_pe_rot (head0)
    ):
        with T.Kernel(1, 1, threads=256) as (bx, by):
            ckv_s = T.alloc_shared([block_N, K_TILE], dtype)
            kvb_w_s = T.alloc_shared([qk_nope, K_TILE], dtype)
            k_nope_s = T.alloc_shared([block_N, qk_nope], dtype)
            v_s = T.alloc_shared([block_N, v_head], dtype)
            k_pe_s = T.alloc_shared([block_N, qk_rope], dtype)
            cos_s = T.alloc_shared([block_N, half], dtype)
            sin_s = T.alloc_shared([block_N, half], dtype)
            sq_local = T.alloc_fragment([block_N, K_TILE], accum)
            sq_sum = T.alloc_fragment([block_N], accum)
            rinv = T.alloc_fragment([block_N], accum)
            kvb_frag = T.alloc_fragment([block_N, qk_nope], accum)
            pe_lo = T.alloc_fragment([block_N, half], accum)
            pe_hi = T.alloc_fragment([block_N, half], accum)

            # rmsnorm sq
            T.clear(sq_local)
            for kk in T.serial(T.ceildiv(kv_lora, K_TILE)):
                T.copy(Latent[0:block_N, 0, kk*K_TILE:(kk+1)*K_TILE], ckv_s)
                for i, j in T.Parallel(block_N, K_TILE):
                    v = T.cast(ckv_s[i, j], accum)
                    sq_local[i, j] += v * v
            T.reduce_sum(sq_local, sq_sum, dim=1)
            for i in T.Parallel(block_N):
                rinv[i] = T.rsqrt(sq_sum[i] / kv_lora + 1e-6)
            # kn
            T.fill(kvb_frag, 0)
            for kk in T.serial(T.ceildiv(kv_lora, K_TILE)):
                T.copy(Latent[0:block_N, 0, kk*K_TILE:(kk+1)*K_TILE], ckv_s)
                for i, j in T.Parallel(block_N, K_TILE):
                    ckv_s[i, j] = T.cast(T.cast(ckv_s[i, j], accum)*rinv[i]*T.cast(kva_ln_w[kk*K_TILE+j], accum), dtype)
                T.copy(kvb_w[0:qk_nope, kk*K_TILE:(kk+1)*K_TILE], kvb_w_s)
                T.gemm(ckv_s, kvb_w_s, kvb_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_N, qk_nope):
                k_nope_s[i, j] = T.cast(kvb_frag[i, j], dtype)
            # v
            T.fill(kvb_frag, 0)
            for kk in T.serial(T.ceildiv(kv_lora, K_TILE)):
                T.copy(Latent[0:block_N, 0, kk*K_TILE:(kk+1)*K_TILE], ckv_s)
                for i, j in T.Parallel(block_N, K_TILE):
                    ckv_s[i, j] = T.cast(T.cast(ckv_s[i, j], accum)*rinv[i]*T.cast(kva_ln_w[kk*K_TILE+j], accum), dtype)
                T.copy(kvb_w[qk_nope:qk_nope+v_head, kk*K_TILE:(kk+1)*K_TILE], kvb_w_s)
                T.gemm(ckv_s, kvb_w_s, kvb_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_N, v_head):
                v_s[i, j] = T.cast(kvb_frag[i, j], dtype)
            # rope
            T.copy(Latent[0:block_N, 0, kv_lora:kv_lora+qk_rope], k_pe_s)
            T.copy(cos_k[0:block_N, :half], cos_s)
            T.copy(sin_k[0:block_N, :half], sin_s)
            for i, j in T.Parallel(block_N, half):
                pe_lo[i, j] = T.cast(k_pe_s[i, 2*j], accum)
                pe_hi[i, j] = T.cast(k_pe_s[i, 2*j+1], accum)
            for i, j in T.Parallel(block_N, half):
                c = T.cast(cos_s[i, j], accum); s = T.cast(sin_s[i, j], accum)
                k_pe_s[i, j] = T.cast(pe_lo[i, j]*c - pe_hi[i, j]*s, dtype)
            for i, j in T.Parallel(block_N, half):
                c = T.cast(cos_s[i, j], accum); s = T.cast(sin_s[i, j], accum)
                k_pe_s[i, half+j] = T.cast(pe_lo[i, j]*s + pe_hi[i, j]*c, dtype)
            # 写出
            for i, j in T.Parallel(block_N, qk_nope):
                Out[i, j] = k_nope_s[i, j]
            for i, j in T.Parallel(block_N, v_head):
                Out[i, qk_nope + j] = v_s[i, j]
            for i, j in T.Parallel(block_N, qk_rope):
                Out[i, qk_nope + v_head + j] = k_pe_s[i, j]
    return main


def main():
    engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
    gr = engine.graph_runner; A = gr.adapter
    blocks = A.blocks(gr.model)
    layer_idx = [i for i, b in enumerate(blocks) if getattr(b.mlp, "_is_moe", False)][0]
    attn = blocks[layer_idx].self_attn

    # 随机构造单 block latent
    torch.manual_seed(1)
    Latent = torch.randn(block_N, 1, kv_lora+qk_rope, device="cuda", dtype=torch.bfloat16)
    kva_ln_w = attn._kva_ln_w.contiguous()
    kvb_w = attn._kvb_w.contiguous()  # [H*kvb_out, kv_lora]
    # cos/sin
    cos_full, sin_full = A._rope_pool(gr, "cuda")
    cos_k = cos_full[:block_N].contiguous()
    sin_k = sin_full[:block_N].contiguous()

    kernel = make_kvb_kernel()
    out = kernel(Latent, kva_ln_w, kvb_w, cos_k, sin_k)  # [block_N, H*(128+128)+64]
    kn_k = out[:, :block_N*0+0]  # placeholder
    kn = out[:, :qk_nope*H].view(block_N, H, qk_nope) if False else out[:, :block_N]
    # 正确切片：Out = [k_nope(block_N,qk_nope) | v(block_N,v_head) | k_pe_rot(block_N,qk_rope)]
    kn_out = out[:, :qk_nope]                  # [block_N, 128] —— 但这是 head 0 的？不对
    # 实际上 kernel 只算了 head 0（kvb_w[0:qk_nope]），Out 布局是 per-row 的单 head
    print("out shape", out.shape)

    # PyTorch 参考
    compressed = Latent[:, 0, :kv_lora]   # [block_N, 512]
    k_pe_raw = Latent[:, 0, kv_lora:]     # [block_N, 64]
    # rmsnorm
    ckv = compressed.float()
    var = ckv.pow(2).mean(-1, keepdim=True)
    ckv_n = ckv * torch.rsqrt(var + 1e-6) * kva_ln_w.float()
    ckv_n = ckv_n.to(torch.bfloat16)
    kv = F.linear(ckv_n, kvb_w).view(block_N, H, qk_nope+v_head)
    k_nope_ref = kv[:, 0, :qk_nope]   # head 0
    v_ref = kv[:, 0, qk_nope:]
    # rope ref
    def apply_rope(x, cos, sin):
        *lead, d = x.shape
        xx = x.reshape(*lead, d//2, 2).transpose(-1,-2).reshape(*lead, d)
        rh = torch.cat((-xx[..., d//2:], xx[..., :d//2]), dim=-1)
        return xx*cos + rh*sin
    kpe_ref = apply_rope(k_pe_raw, cos_k, sin_k)

    # kernel 输出：head 0 的 k_nope 在 Out[:, :qk_nope]，v 在 [:, qk_nope:qk_nope+v_head]
    kn_k = out[:, :qk_nope]
    v_k = out[:, qk_nope:qk_nope+v_head]
    kpe_k = out[:, qk_nope+v_head:]

    print("k_nope max_diff:", (kn_k.float()-k_nope_ref.float()).abs().max().item())
    print("v      max_diff:", (v_k.float()-v_ref.float()).abs().max().item())
    print("k_pe   max_diff:", (kpe_k.float()-kpe_ref.float()).abs().max().item())


if __name__ == "__main__":
    main()
