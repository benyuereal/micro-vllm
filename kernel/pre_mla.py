"""
MLA 前置全融合 kernel：q_proj(+rope) / kva_proj(+store) / absorb。

把 MLA attention kernel 之前的零碎 PyTorch 算子（q_proj、kva_proj、store latent、
rope(q_pe)、einsum absorb）融进 kernel，消除 bs=1 下 pre-MLA 的 execution gap。

为什么是 3 个 kernel 而非 1 个：q_proj 与 absorb 是串行 GEMM（absorb 需要 q_proj 输出
q_nope），非 persistent kernel 无法跨 block 同步，单 grid 内 GEMM→依赖 GEMM
会竞争。评估过"融进每个 MLA split（4× 冗余）"——q_proj 100M MACs ×4 = 545M，是
attention loop 的 122×，反而更慢。故拆成 2 个紧耦合 pre-kernel + absorb。

- pre_qkv: q_proj GEMM + rope epilogue（q_pe 列）。输出 [bs, H, 16, q_head]。
- pre_kva: kva_proj GEMM + store epilogue（latent 直写 paged cache）。
- absorb:  q_nope @ kvb_w_kn_t → A（复用已验证的 M=16 per-head GEMV）。

M=16 零填充（mma.h 要求 M%16==0），只读 row 0 真实数据。bf16，fp32 累加。
"""
import torch
import tilelang
import tilelang.language as T


_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}


@tilelang.jit(
    out_idx=[5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def pre_qkv_kernel(bs, hidden, h_attn, q_head, qk_rope, dtype):
    """q_proj GEMM + rope(q_pe) epilogue。

    grid=(bs, cdiv(Q_OUT,64))。每 block 算 q 的 64 输出列，M=16 pad 只用 row 0。
    输出 O[bs, H, 16, q_head]：epilogue 里 h=col//q_head, d=col%q_head 写 O[b,h,0,d]，
    使 q_nope=O[:,:,:,:qk_nope] 对 absorb 直接可喂。
    rope 只在 (nblk % q_head_blocks)==(q_head_blocks-1) 的 block（每 head 末个 64-block=q_pe）。
    rope 精确复刻 adapter _apply_rope（=HF DeepSeek view_as_complex 复数乘的等价实数展开）：
    对每对 (qpe[2k], qpe[2k+1]) 与 freqs_cis[k]，输出 **deinterleaved** 布局——
    out[k]=a*cos-b*sin 写列 128+k，out[k+half]=a*sin+b*cos 写列 128+k+half。
    注意：输出是 deinterleaved（不是原 interleaved），与 _apply_rope / MLA kernel 约定一致。
    """
    accum = T.float32
    Q_OUT = h_attn * q_head
    half = qk_rope // 2
    q_head_blocks = q_head // 64  # 192/64 = 3

    @T.prim_func
    def main(
        X16: T.Tensor([bs, 16, hidden], dtype),
        W: T.Tensor([Q_OUT, hidden], dtype),
        B: T.Tensor([Q_OUT], dtype),
        Cos: T.Tensor([bs, qk_rope], dtype),
        Sin: T.Tensor([bs, qk_rope], dtype),
        O: T.Tensor([bs, h_attn, 16, q_head], dtype),
    ):
        with T.Kernel(bs, T.ceildiv(Q_OUT, 64), threads=128) as (b, nblk):
            X_s = T.alloc_shared([16, 128], dtype)
            W_s = T.alloc_shared([64, 128], dtype)
            acc = T.alloc_fragment([16, 64], accum)
            acc_s = T.alloc_shared([16, 64], dtype)
            T.clear(acc)
            for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                T.copy(X16[b, 0:16, kh * 128:(kh + 1) * 128], X_s)
                T.copy(W[nblk * 64:(nblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, acc_s)
            is_pe = (nblk % q_head_blocks) == (q_head_blocks - 1)
            cs = T.alloc_shared([qk_rope], dtype)
            ss = T.alloc_shared([qk_rope], dtype)
            for j in T.Parallel(qk_rope):
                cs[j] = Cos[b, j]
                ss[j] = Sin[b, j]
            if is_pe:
                # interleaved RoPE → deinterleaved 输出（与 adapter _apply_rope 完全一致，
                # 即 HF DeepSeek view_as_complex 复数乘法的等价实数展开）。
                # 对每对 (a=qpe[2k], b=qpe[2k+1])，freqs_cis[k]=cos(θ_k)+i·sin(θ_k)：
                #   out[k]      = a*cos - b*sin   → 写列 128+k
                #   out[k+half] = a*sin + b*cos   → 写列 128+k+half
                # cos/sin 全宽 cat(freqs,freqs)，故 cs[k]==cs[k+half]，用 cs[k] 即可。
                for k in T.Parallel(half):
                    pa = T.cast(acc_s[0, 2 * k], accum) + T.cast(B[nblk * 64 + 2 * k], accum)
                    pb = T.cast(acc_s[0, 2 * k + 1], accum) + T.cast(B[nblk * 64 + 2 * k + 1], accum)
                    ck = T.cast(cs[k], accum)
                    sk = T.cast(ss[k], accum)
                    c0 = nblk * 64 + k
                    c1 = nblk * 64 + k + half
                    h0 = c0 // q_head; d0 = c0 % q_head
                    h1 = c1 // q_head; d1 = c1 % q_head
                    O[b, h0, 0, d0] = T.cast(pa * ck - pb * sk, dtype)
                    O[b, h1, 0, d1] = T.cast(pa * sk + pb * ck, dtype)
            else:
                for j in T.Parallel(64):
                    val = T.cast(acc_s[0, j], accum) + T.cast(B[nblk * 64 + j], accum)
                    c = nblk * 64 + j
                    h = c // q_head
                    d = c % q_head
                    O[b, h, 0, d] = T.cast(val, dtype)
    return main


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def pre_kva_kernel(bs, hidden, kva_out, block_size, max_seq_blocks, n_blocks, dtype):
    """kva_proj GEMM + store latent 到 paged cache 的 epilogue（k_cache 与 v_cache 同 latent）。

    grid=(bs, cdiv(kva_out,64))。每 block 算 kva 的 64 列，epilogue 直写 k_cache 与 v_cache
    同一 slot：blk_id = block_table[b, new_pos//block_size]，offset = new_pos%block_size，
    K_cache[blk_id, offset, 0, col] = val。消除独立 store launch。
    MLA 只读 k_cache（reshape 成 [n_blocks*block_size, 1, 576]），v_cache 写入仅为框架契约一致。
    """
    accum = T.float32

    @T.prim_func
    def main(
        X16: T.Tensor([bs, 16, hidden], dtype),
        W: T.Tensor([kva_out, hidden], dtype),
        B: T.Tensor([kva_out], dtype),
        Block_table: T.Tensor([bs, max_seq_blocks], T.int32),
        New_pos: T.Tensor([bs], T.int32),
        K_cache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
        V_cache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
    ):
        with T.Kernel(bs, T.ceildiv(kva_out, 64), threads=128) as (b, nblk):
            X_s = T.alloc_shared([16, 128], dtype)
            W_s = T.alloc_shared([64, 128], dtype)
            acc = T.alloc_fragment([16, 64], accum)
            acc_s = T.alloc_shared([16, 64], dtype)
            T.clear(acc)
            for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                T.copy(X16[b, 0:16, kh * 128:(kh + 1) * 128], X_s)
                T.copy(W[nblk * 64:(nblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, acc_s)
            pos = New_pos[b]
            blk_id = Block_table[b, pos // block_size]
            offset = pos % block_size
            for j in T.Parallel(64):
                val = T.cast(T.cast(acc_s[0, j], accum) + T.cast(B[nblk * 64 + j], accum), dtype)
                K_cache[blk_id, offset, 0, nblk * 64 + j] = val
                V_cache[blk_id, offset, 0, nblk * 64 + j] = val
    return main


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def absorb_kernel(batch, h_attn, qk_nope, kv_lora, dtype):
    """absorb: q_nope[b,h,d] @ kvb_w_kn_t[h,k,d] → A[b,h,k]（M=16 per-head GEMV）。

    grid=(batch*H, cdiv(kv_lora,64))。W_kn 输入需转置 [H, kv_lora, qk_nope]。
    输出 [batch*H, kv_lora]（M=16 仅内部 GEMM 用，输出丢掉 pad 维），reshape 成
    [batch, H, kv_lora] 后是 contiguous，stride[1]=kv_lora，符合 MLA kernel 输入要求。
    """
    accum = T.float32

    @T.prim_func
    def main(
        Q_nope16: T.Tensor([batch * h_attn, 16, qk_nope], dtype),
        W_kn: T.Tensor([h_attn, kv_lora, qk_nope], dtype),
        IDX: T.Tensor([batch * h_attn], T.int32),
        A_out: T.Tensor([batch * h_attn, kv_lora], dtype),
    ):
        with T.Kernel(batch * h_attn, T.ceildiv(kv_lora, 64), threads=128) as (bh, kblk):
            X_s = T.alloc_shared([16, 128], dtype)
            W_s = T.alloc_shared([64, 128], dtype)
            acc = T.alloc_fragment([16, 64], accum)
            acc_s = T.alloc_shared([16, 64], dtype)
            h = IDX[bh]
            T.clear(acc)
            for kh in T.Pipelined(T.ceildiv(qk_nope, 128), num_stages=2):
                T.copy(Q_nope16[bh, 0:16, kh * 128:(kh + 1) * 128], X_s)
                T.copy(W_kn[h, kblk * 64:(kblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, acc_s)
            for j in T.Parallel(64):
                A_out[bh, kblk * 64 + j] = T.cast(acc_s[0, j], dtype)
    return main


# -------------------- launcher / cache --------------------
_pre_qkv_cache = {}
_pre_kva_cache = {}
_absorb_cache = {}


def get_pre_qkv_kernel(bs, hidden, h_attn, q_head, qk_rope, dtype):
    key = (bs, hidden, h_attn, q_head, qk_rope, dtype)
    if key not in _pre_qkv_cache:
        _pre_qkv_cache[key] = pre_qkv_kernel(bs, hidden, h_attn, q_head, qk_rope, _TORCH_TO_TL[dtype])
    return _pre_qkv_cache[key]


def get_pre_kva_kernel(bs, hidden, kva_out, block_size, max_seq_blocks, n_blocks, dtype):
    key = (bs, hidden, kva_out, block_size, max_seq_blocks, n_blocks, dtype)
    if key not in _pre_kva_cache:
        _pre_kva_cache[key] = pre_kva_kernel(bs, hidden, kva_out, block_size, max_seq_blocks, n_blocks, _TORCH_TO_TL[dtype])
    return _pre_kva_cache[key]


def get_absorb_kernel(batch, h_attn, qk_nope, kv_lora, dtype):
    key = (batch, h_attn, qk_nope, kv_lora, dtype)
    if key not in _absorb_cache:
        _absorb_cache[key] = absorb_kernel(batch, h_attn, qk_nope, kv_lora, _TORCH_TO_TL[dtype])
    return _absorb_cache[key]
