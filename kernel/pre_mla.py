"""
MLA 前置全融合 persistent kernel：pre_qkv ∥ pre_kva → absorb 单 kernel。

把 MLA attention kernel 之前的零碎 PyTorch 算子（q_proj、kva_proj、store latent、
rope(q_pe)、einsum absorb）融进单个 persistent kernel，消除 bs=1 下 pre-MLA 的
execution gap（graph 路径 +2.0%，见 premla-persistent-roi）。

3-phase（T.Kernel(NUM_SMS=92)，phase 间 T.sync_grid 屏障）：
  phase1: pre_qkv(x16,q_w,rope) → q_out  ∥  pre_kva(x16,kva_w) → k/v_cache
          （两段并行都读 x16，用 task id 区分）
  phase2: absorb(q_nope,kvb_kn) → A

- pre_qkv: q_proj GEMM + rope epilogue（q_pe 列），直写紧凑 QpeOut。
- pre_kva: kva_proj GEMM + store epilogue（latent 直写 paged cache）。
- absorb:  q_nope @ kvb_w_kn_t → A（M=16 per-head GEMV）。

X16 row0 由调用方 rmsnorm_ 预填（graph._x16[:bs,0,:]），rows1-15 恒零。
M=16 零填充（mma.h 要求 M%16==0），只读 row 0 真实数据。bf16，fp32 累加。
"""
import torch
import tilelang
import tilelang.language as T


_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}
NUM_SMS = 92


# ============ persistent kernel: pre_qkv ∥ pre_kva + absorb 单 kernel ============
# 3-phase persistent kernel，phase 间 T.sync_grid 屏障，中间 x16/q_out 经全局 buffer 通信。
# 跨依赖 kernel 链的 persistent 在 graph 路径下吃 execution gap（+2.0% ROI，见 premla-persistent-roi）。
@tilelang.jit(
    out_idx=[11],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def premla_persistent_kernel(
    bs, hidden, h_attn, q_head, qk_rope, qk_nope, kv_lora, kva_out,
    block_size, max_seq_blocks, n_blocks, max_pos, dtype,
):
    """3-phase persistent: pre_qkv(144)∥pre_kva(9) → absorb(128)。
    输出 A[bs*h_attn, kv_lora]（out_idx=10）。X16 由调用方 rmsnorm_ 预填 row0（graph._x16[:bs]），
    QOut 需预分配传入。bs>1 路径暂不支持（kernel 内硬编码 [0,...]，仅 bs=1 ROI 验证用）。
    Cos/Sin 传全池 [max_pos, qk_rope]，kernel 内部按 NewPos gather（省外部 cos[new_pos] gather+cast）。"""
    accum = T.float32
    Q_OUT = h_attn * q_head
    half = qk_rope // 2
    q_head_blocks = q_head // 64
    N_QKV = T.ceildiv(Q_OUT, 64)
    N_KVA = T.ceildiv(kva_out, 64)
    N_ABS = h_attn * T.ceildiv(kv_lora, 64)

    @T.prim_func
    def main(
        QW: T.Tensor([Q_OUT, hidden], dtype),
        QB: T.Tensor([Q_OUT], dtype),
        Cos: T.Tensor([max_pos, qk_rope], dtype),
        Sin: T.Tensor([max_pos, qk_rope], dtype),
        KvaW: T.Tensor([kva_out, hidden], dtype),
        KvaB: T.Tensor([kva_out], dtype),
        KvbKn: T.Tensor([h_attn, kv_lora, qk_nope], dtype),
        AbsIdx: T.Tensor([bs * h_attn], T.int32),
        X16: T.Tensor([bs, 16, hidden], dtype),
        QOut: T.Tensor([bs, h_attn, 16, q_head], dtype),
        QpeOut: T.Tensor([bs, h_attn, qk_rope], dtype),
        AOut: T.Tensor([bs * h_attn, kv_lora], dtype),
        BlockTable: T.Tensor([bs, max_seq_blocks], T.int32),
        NewPos: T.Tensor([bs], T.int32),
        Kcache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
        Vcache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
    ):
        with T.Kernel(NUM_SMS, threads=128) as (sm_idx,):
            # X16 row0 已由调用方 rmsnorm_ 预填（graph._x16[:bs,0,:]），rows1-15 恒零（alloc 时 zeros）。
            # ===== phase1: pre_qkv(144) ∥ pre_kva(9) =====
            N_TOTAL = N_QKV + N_KVA
            for task in T.serial(sm_idx, N_TOTAL, NUM_SMS):
                if task < N_QKV:
                    nblk = task
                    X_s = T.alloc_shared([16, 128], dtype)
                    W_s = T.alloc_shared([64, 128], dtype)
                    acc = T.alloc_fragment([16, 64], accum)
                    acc_s = T.alloc_shared([16, 64], dtype)
                    T.clear(acc)
                    for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                        T.copy(X16[0, 0:16, kh * 128:(kh + 1) * 128], X_s)
                        T.copy(QW[nblk * 64:(nblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                        T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(acc, acc_s)
                    is_pe = (nblk % q_head_blocks) == (q_head_blocks - 1)
                    cs = T.alloc_shared([qk_rope], dtype)
                    ss = T.alloc_shared([qk_rope], dtype)
                    # kernel 内部按 NewPos[0] 从全池 gather cos/sin（省外部 cos[new_pos].to(dtype)）
                    qpos = NewPos[0]
                    for j in T.Parallel(qk_rope):
                        cs[j] = Cos[qpos, j]; ss[j] = Sin[qpos, j]
                    if is_pe:
                        for k in T.Parallel(half):
                            pa = T.cast(acc_s[0, 2 * k], accum) + T.cast(QB[nblk * 64 + 2 * k], accum)
                            pb = T.cast(acc_s[0, 2 * k + 1], accum) + T.cast(QB[nblk * 64 + 2 * k + 1], accum)
                            ck = T.cast(cs[k], accum); sk = T.cast(ss[k], accum)
                            c0 = nblk * 64 + k; c1 = nblk * 64 + k + half
                            h0 = c0 // q_head; h1 = c1 // q_head
                            v0 = T.cast(pa * ck - pb * sk, dtype)
                            v1 = T.cast(pa * sk + pb * ck, dtype)
                            QOut[0, h0, 0, c0 % q_head] = v0
                            QOut[0, h1, 0, c1 % q_head] = v1
                            # 同时写紧凑 QpeOut[bs, h, qk_rope]（contiguous，供 MLA 直接读，省外部 slice+contiguous）
                            QpeOut[0, h0, k] = v0
                            QpeOut[0, h1, k + half] = v1
                    else:
                        for j in T.Parallel(64):
                            val = T.cast(acc_s[0, j], accum) + T.cast(QB[nblk * 64 + j], accum)
                            c = nblk * 64 + j
                            QOut[0, c // q_head, 0, c % q_head] = T.cast(val, dtype)
                else:
                    nblk = task - N_QKV
                    X_s = T.alloc_shared([16, 128], dtype)
                    W_s = T.alloc_shared([64, 128], dtype)
                    acc = T.alloc_fragment([16, 64], accum)
                    acc_s = T.alloc_shared([16, 64], dtype)
                    T.clear(acc)
                    for kh in T.Pipelined(T.ceildiv(hidden, 128), num_stages=2):
                        T.copy(X16[0, 0:16, kh * 128:(kh + 1) * 128], X_s)
                        T.copy(KvaW[nblk * 64:(nblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                        T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(acc, acc_s)
                    pos = NewPos[0]
                    blk_id = BlockTable[0, pos // block_size]
                    offset = pos % block_size
                    for j in T.Parallel(64):
                        val = T.cast(T.cast(acc_s[0, j], accum) + T.cast(KvaB[nblk * 64 + j], accum), dtype)
                        Kcache[blk_id, offset, 0, nblk * 64 + j] = val
                        Vcache[blk_id, offset, 0, nblk * 64 + j] = val
            T.sync_grid()
            # ===== phase2: absorb (128 task) =====
            for task in T.serial(sm_idx, N_ABS, NUM_SMS):
                bh = task // T.ceildiv(kv_lora, 64)
                kblk = task % T.ceildiv(kv_lora, 64)
                h = AbsIdx[bh]
                X_s = T.alloc_shared([16, 128], dtype)
                W_s = T.alloc_shared([64, 128], dtype)
                acc = T.alloc_fragment([16, 64], accum)
                acc_s = T.alloc_shared([16, 64], dtype)
                T.clear(acc)
                for kh in T.Pipelined(T.ceildiv(qk_nope, 128), num_stages=2):
                    T.copy(QOut[0, h, 0:16, kh * 128:(kh + 1) * 128], X_s)
                    T.copy(KvbKn[h, kblk * 64:(kblk + 1) * 64, kh * 128:(kh + 1) * 128], W_s)
                    T.gemm(X_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(acc, acc_s)
                for j in T.Parallel(64):
                    AOut[bh, kblk * 64 + j] = T.cast(acc_s[0, j], dtype)
    return main


# -------------------- launcher / cache --------------------
_premla_persist_cache = {}


def get_premla_persistent_kernel(bs, hidden, h_attn, q_head, qk_rope, qk_nope, kv_lora,
                                 kva_out, block_size, max_seq_blocks, n_blocks, max_pos, dtype):
    key = (bs, hidden, h_attn, q_head, qk_rope, qk_nope, kv_lora, kva_out,
           block_size, max_seq_blocks, n_blocks, max_pos, dtype)
    if key not in _premla_persist_cache:
        _premla_persist_cache[key] = (
            premla_persistent_kernel(bs, hidden, h_attn, q_head, qk_rope, qk_nope, kv_lora,
                                     kva_out, block_size, max_seq_blocks, n_blocks, max_pos,
                                     _TORCH_TO_TL[dtype]),
            torch.empty(bs, h_attn, 16, q_head, dtype=dtype, device="cuda"),    # QOut 中间（absorb 读 q_nope）
            torch.empty(bs, h_attn, qk_rope, dtype=dtype, device="cuda"),       # QpeOut 紧凑（MLA 读）
        )
    return _premla_persist_cache[key]
