"""
Fused MLA decode kernel：latent → rmsnorm + RoPE + paged flash 全融合（weight-absorption）。

把当前 attention 的 gather(15)+kvb(47)+rope(24)+cat/pad(44)+flash(8) = 138us
压成一个 kernel，中间的 [bs,1024,16,256] 全程不落 HBM。

**关键：DeepSeek kv_b_proj 是 per-head 的**（kvb_w [H*256, 512]，每 head 独立 k_nope/v），
不能像 example 那样跨 head 共享 KV。用 MLA weight-absorption 把 per-head kvb 权重吸收：
- 预算（adapter 里）：A[h] = Q_nope[h] @ kvb_w[h*256 : h*256+128]  → A[H, kv_lora]（每层一次）
- flash 循环内：QK = A @ ckv_norm^T + Q_pe @ k_pe_rot^T（标准 gemm，无 per-head KV）
                softmax；P += softmax @ ckv_norm（累加到 [H, kv_lora] 空间）
- combine：P_global 跨 split 加权后 out[h] = P_global[h] @ kvb_w_v[h]^T（每层一次）
这样 per-head 的两个 einsum 各做一次，flash 内全是标准 gemm。

block_size=256, block_N=64, K_TILE=128, bf16。
smem（L20 上限 100KB）：A_s[16,128]=4KB, ckv_s[64,128]=16KB(复用存 k_pe), Q_pe_s[16,64]=2KB,
  cos/sin[64,32]x2=8KB, S_s[16,64]=2KB → ~32KB，余量充足。
register：acc_p[H, kv_lora]=[16,512] fp32 = 32 regs/thread；acc_s/acc_s_cast 等。
"""
import torch
import tilelang
import tilelang.language as T


_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}


@tilelang.jit(
    out_idx=[9],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def fused_mla_decode_kernel(
    batch, h_q, max_seqlen, kv_lora, qk_rope, qk_nope, v_head,
    block_N, num_split, block_size, softmax_scale, dtype, n_slots,
):
    """Fused MLA decode（split-KV, weight-absorption）。每 program = (batch, split)，所有 H 在一个 program 内。"""
    accum_dtype = T.float32
    half_qk_rope = qk_rope // 2             # 32
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    K_TILE = 128
    assert kv_lora % K_TILE == 0
    BLOCK_H = h_q  # V2-Lite: 16

    @T.prim_func
    def main_split(
        A: T.Tensor([batch, h_q, kv_lora], dtype),              # 吸收后的 Q: Q_nope @ kvb_w_kn
        Q_pe: T.Tensor([batch, h_q, qk_rope], dtype),           # 已 RoPE
        Latent: T.Tensor([n_slots, 1, kv_lora + qk_rope], dtype),
        block_table: T.Tensor([batch, max_seqlen // block_size], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        kva_ln_w: T.Tensor([kv_lora], dtype),
        kvb_w_v: T.Tensor([h_q, v_head, kv_lora], dtype),       # post-multiply 用（per-head v 权重）
        cos_k: T.Tensor([max_seqlen, qk_rope], dtype),
        sin_k: T.Tensor([max_seqlen, qk_rope], dtype),
        Output: T.Tensor([batch, h_q, v_head], dtype),
    ):
        glse: T.Tensor([batch, h_q, num_split], dtype) = T.alloc_buffer([batch, h_q, num_split], dtype)
        P_partial: T.Tensor([batch, h_q, num_split, kv_lora], dtype) = T.alloc_buffer([batch, h_q, num_split, kv_lora], dtype)

        with T.Kernel(batch, 1, num_split, threads=256) as (bx, by, bz):
            A_s = T.alloc_shared([BLOCK_H, K_TILE], dtype)      # A 分 K_TILE 加载
            Q_pe_s = T.alloc_shared([BLOCK_H, qk_rope], dtype)
            ckv_s = T.alloc_shared([block_N, K_TILE], dtype)    # rmsnorm/QK/P 用；RoPE 阶段复用存 k_pe
            k_pe_s = T.alloc_shared([block_N, qk_rope], dtype)
            cos_s = T.alloc_shared([block_N, half_qk_rope], dtype)
            sin_s = T.alloc_shared([block_N, half_qk_rope], dtype)
            S_s = T.alloc_shared([BLOCK_H, block_N], dtype)
            # [BLOCK_H] 标量状态全部走 shared，避免被 reduce/gemm 推断出冲突的 fragment layout
            scale_s = T.alloc_shared([BLOCK_H], accum_dtype)  # scores_scale（fp32 保精度）
            sum_s = T.alloc_shared([BLOCK_H], accum_dtype)    # scores_sum（reduce 输出直写 shared）
            max_s = T.alloc_shared([BLOCK_H], accum_dtype)    # scores_max (running, reduce 输出直写 shared)
            max_prev_s = T.alloc_shared([BLOCK_H], accum_dtype)  # scores_max_prev
            logsum_s = T.alloc_shared([BLOCK_H], accum_dtype) # logsum
            acc_s = T.alloc_fragment([BLOCK_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([BLOCK_H, block_N], dtype)
            acc_p = T.alloc_shared([BLOCK_H, kv_lora], accum_dtype)  # 累加到 kv_lora 空间（shared，跨 K_TILE 切片累加安全）
            # rmsnorm tiled
            sq_local = T.alloc_fragment([block_N, K_TILE], accum_dtype)
            sq_sum = T.alloc_fragment([block_N], accum_dtype)
            rinv = T.alloc_fragment([block_N], accum_dtype)
            # k_pe RoPE
            pe_lo = T.alloc_fragment([block_N, half_qk_rope], accum_dtype)
            pe_hi = T.alloc_fragment([block_N, half_qk_rope], accum_dtype)
            # P 累加时复用的 softmax@ckv 临时 fragment（K_TILE 分块）
            p_frag = T.alloc_fragment([BLOCK_H, K_TILE], accum_dtype)

            T.use_swizzle(10)
            T.copy(Q_pe[bx, :, :], Q_pe_s)
            T.fill(acc_p, 0)
            for i in T.Parallel(BLOCK_H):
                logsum_s[i] = 0
                max_s[i] = -T.infinity(accum_dtype)

            total_blocks = T.ceildiv(cache_seqlens[bx], block_N)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
            start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N

            for k in T.serial(loop_range):
                kv_logical = start + k * block_N
                kv_start = block_table[bx, kv_logical // block_size] * block_size + (kv_logical % block_size)

                # ---- rmsnorm(compressed_kv): 累加各 K_TILE 平方到 sq_local，reduce_sum ----
                T.clear(sq_local)
                for kk1 in T.serial(T.ceildiv(kv_lora, K_TILE)):
                    T.copy(Latent[kv_start:kv_start + block_N, 0, kk1 * K_TILE:(kk1 + 1) * K_TILE], ckv_s)
                    for i, j in T.Parallel(block_N, K_TILE):
                        v = T.cast(ckv_s[i, j], accum_dtype)
                        sq_local[i, j] += v * v
                T.reduce_sum(sq_local, sq_sum, dim=1)
                for i in T.Parallel(block_N):
                    rinv[i] = T.rsqrt(sq_sum[i] / kv_lora + 1e-6)

                # ---- QK = A @ ckv_norm^T + Q_pe @ k_pe_rot^T ----
                # 第一项 K_TILE 分块：acc_s = sum_kk A_s[:,:,kk] @ ckv_norm[:,:,kk]^T
                T.clear(acc_s)
                for kk2 in T.serial(T.ceildiv(kv_lora, K_TILE)):
                    T.copy(A[bx, :, kk2 * K_TILE:(kk2 + 1) * K_TILE], A_s)
                    T.copy(Latent[kv_start:kv_start + block_N, 0, kk2 * K_TILE:(kk2 + 1) * K_TILE], ckv_s)
                    # ckv_s = ckv * rinv * kva_ln_w（rmsnorm 输出，就地）
                    for i, j in T.Parallel(block_N, K_TILE):
                        ckv_s[i, j] = T.cast(T.cast(ckv_s[i, j], accum_dtype) * rinv[i] * T.cast(kva_ln_w[kk2 * K_TILE + j], accum_dtype), dtype)
                    T.gemm(A_s, ckv_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                # 第二项：k_pe RoPE 后 Q_pe @ k_pe_rot^T
                T.copy(Latent[kv_start:kv_start + block_N, 0, kv_lora:kv_lora + qk_rope], ckv_s)
                T.copy(cos_k[kv_logical:kv_logical + block_N, :half_qk_rope], cos_s)
                T.copy(sin_k[kv_logical:kv_logical + block_N, :half_qk_rope], sin_s)
                for i, j in T.Parallel(block_N, half_qk_rope):
                    pe_lo[i, j] = T.cast(ckv_s[i, 2 * j], accum_dtype)
                    pe_hi[i, j] = T.cast(ckv_s[i, 2 * j + 1], accum_dtype)
                for i, j in T.Parallel(block_N, half_qk_rope):
                    c = T.cast(cos_s[i, j], accum_dtype)
                    s = T.cast(sin_s[i, j], accum_dtype)
                    k_pe_s[i, j] = T.cast(pe_lo[i, j] * c - pe_hi[i, j] * s, dtype)
                for i, j in T.Parallel(block_N, half_qk_rope):
                    c = T.cast(cos_s[i, j], accum_dtype)
                    s = T.cast(sin_s[i, j], accum_dtype)
                    k_pe_s[i, half_qk_rope + j] = T.cast(pe_lo[i, j] * s + pe_hi[i, j] * c, dtype)
                T.gemm(Q_pe_s, k_pe_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                # ---- online softmax（标量状态全部 shared）----
                T.copy(max_s, max_prev_s)
                for i in T.Parallel(BLOCK_H):
                    max_s[i] = -T.infinity(accum_dtype)
                for i, j in T.Parallel(BLOCK_H, block_N):
                    acc_s[i, j] = T.if_then_else(kv_logical + j >= cache_seqlens[bx], -T.infinity(accum_dtype), acc_s[i, j])
                T.reduce_max(acc_s, max_s, dim=1, clear=False)
                for i in T.Parallel(BLOCK_H):
                    max_s[i] = T.max(max_s[i], max_prev_s[i])
                for i in T.Parallel(BLOCK_H):
                    scale_s[i] = T.exp2(max_prev_s[i] * scale - max_s[i] * scale)
                for i, j in T.Parallel(BLOCK_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - max_s[i] * scale)
                T.reduce_sum(acc_s, sum_s, dim=1)
                T.copy(acc_s, S_s)
                T.copy(S_s, acc_s_cast)
                for i in T.Parallel(BLOCK_H):
                    logsum_s[i] = logsum_s[i] * scale_s[i] + sum_s[i]
                # rescale acc_p（历史 P 乘 scores_scale；从 shared 读）
                for i, j in T.Parallel(BLOCK_H, kv_lora):
                    acc_p[i, j] *= scale_s[i]

                # ---- P += softmax @ ckv_norm（K_TILE 分块，重算 ckv_norm）----
                for kk3 in T.serial(T.ceildiv(kv_lora, K_TILE)):
                    T.copy(Latent[kv_start:kv_start + block_N, 0, kk3 * K_TILE:(kk3 + 1) * K_TILE], ckv_s)
                    for i, j in T.Parallel(block_N, K_TILE):
                        ckv_s[i, j] = T.cast(T.cast(ckv_s[i, j], accum_dtype) * rinv[i] * T.cast(kva_ln_w[kk3 * K_TILE + j], accum_dtype), dtype)
                    T.fill(p_frag, 0)
                    T.gemm(acc_s_cast, ckv_s, p_frag, policy=T.GemmWarpPolicy.FullCol)
                    for i, j in T.Parallel(BLOCK_H, K_TILE):
                        acc_p[i, kk3 * K_TILE + j] += p_frag[i, j]

            # ---- 输出 P_partial = acc_p / logsum（空 split 守卫）----
            for i, j in T.Parallel(BLOCK_H, kv_lora):
                acc_p[i, j] = T.if_then_else(logsum_s[i] > 0, acc_p[i, j] / logsum_s[i], 0)
            for i in T.Parallel(BLOCK_H):
                logsum_s[i] = T.log2(logsum_s[i]) + max_s[i] * scale
            T.copy(logsum_s, glse[bx, :, bz])
            T.copy(acc_p, P_partial[bx, :, bz, :])

        # combine：跨 split 加权 P_partial → P_global → out = P_global @ kvb_w_v^T（per head）
        # M=16 零填充 T.gemm：p_accum[1,kv_lora] pad 成 [16,kv_lora] 用 tensor core（手写 GEMV 68us→21us/层）
        # P_s/W_s 均 fp32（kvb_w_v cast 上来），全程 fp32 不丢精度；27 层累积下 bf16 量化会退化。
        with T.Kernel(batch, h_q, T.ceildiv(v_head, 64), threads=128) as (bb, bh, vblk):
            p_local = T.alloc_fragment([kv_lora], dtype)
            p_accum = T.alloc_fragment([kv_lora], accum_dtype)
            p_acc_s = T.alloc_shared([kv_lora], accum_dtype)  # p_accum→shared 全宽，供 kk 切片
            P_s = T.alloc_shared([16, 128], accum_dtype)      # fp32，pad 至 16 行，row 0 真实
            W_s = T.alloc_shared([128, 64], accum_dtype)      # fp32，kvb_w_v[bh,..] cast 上来转置
            acc = T.alloc_fragment([16, 64], accum_dtype)
            acc_s = T.alloc_shared([16, 64], accum_dtype)     # fragment→shared 中转，规避 layout 冲突
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(p_accum)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bb, bh, k])
            T.clear(lse_logsum_local)
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bb, bh, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(kv_lora):
                    p_local[i] = P_partial[bb, bh, k, i]
                lse_local_split = glse[bb, bh, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(kv_lora):
                    p_accum[i] += T.cast(p_local[i], accum_dtype) * scale_local
            # out[bh] = p_accum @ kvb_w_v[bh]^T —— M=16 pad T.gemm（取 row 0），K_TILE=128 分块累加
            T.copy(p_accum, p_acc_s)
            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(kv_lora, 128), num_stages=2):
                for i in T.Parallel(128):
                    P_s[0, i] = p_acc_s[kk * 128 + i]
                for i, j in T.Parallel(128, 64):
                    W_s[i, j] = T.cast(kvb_w_v[bh, vblk * 64 + j, kk * 128 + i], accum_dtype)
                T.gemm(P_s, W_s, acc, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, acc_s)
            for j in T.Parallel(64):
                Output[bb, bh, vblk * 64 + j] = T.cast(acc_s[0, j], dtype)

    return main_split


# -------------------- launcher / cache --------------------
_kernel_cache = {}


def _get_kernel(batch, h_q, max_seqlen, kv_lora, qk_rope, qk_nope, v_head,
                block_size, softmax_scale, dtype, n_slots, block_N=64, num_split=4):
    key = (batch, h_q, max_seqlen, kv_lora, qk_rope, qk_nope, v_head,
           block_size, float(softmax_scale), dtype, n_slots, block_N, num_split)
    if key not in _kernel_cache:
        tl_dt = _TORCH_TO_TL[dtype]
        _kernel_cache[key] = fused_mla_decode_kernel(
            batch, h_q, max_seqlen, kv_lora, qk_rope, qk_nope, v_head,
            block_N, num_split, block_size, softmax_scale, tl_dt, n_slots,
        )
    return _kernel_cache[key]
