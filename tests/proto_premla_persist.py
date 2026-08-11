"""pre-MLA 段 persistent kernel 早期原型（ROI 验证用，含 phase0 rmsnorm）。

这是验证"pre_qkv∥pre_kva→absorb 跨依赖链 persistent 在 graph 路径有正 ROI"的早期原型。
live 版本已沉淀进 kernel/pre_mla.py（3-phase，rmsnorm 外置由调用方 rmsnorm_ 预填 x16，
新增 QpeOut/max_pos），本文件保留作历史参考——其内部 kernel 与 live 版已分叉。

融合: rmsnorm → pre_qkv ∥ pre_kva → absorb。
phase0: rmsnorm(h) → x16[0,:]           [1 task, SM0]
sync_grid
phase1: pre_qkv(x16,q_w,rope) → q_out   [144 task]  ∥
phase1: pre_kva(x16,kva_w) → k/v_cache  [9 task]     两段并行(都读 x16)
  → 用 task id 区分: task<144 是 qkv, 144≤task<153 是 kva
sync_grid
phase2: absorb(q_nope,kvb_kn) → A       [128 task]
"""
import sys, torch
sys.path.insert(0, "/models/micro-vllm")
import tilelang
import tilelang.language as T

NUM_SMS = 92
_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}


@tilelang.jit(
    out_idx=[12],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def premla_persistent_kernel(
    bs, hidden, h_attn, q_head, qk_rope, qk_nope, kv_lora, kva_out,
    block_size, max_seq_blocks, n_blocks, dtype,
):
    """pre-MLA 4-phase persistent kernel。
    输入: H_in[bs,hidden], InLnW[hidden], QW[q_out,hidden], QB[q_out],
          Cos[bs,qk_rope], Sin[bs,qk_rope], KvaW[kva_out,hidden], KvaB[kva_out],
          BlockTable[bs,max_seq_blocks], NewPos[bs], Kcache[n_blocks,bs_block,1,kva_out],
          Q_out(中间,预分配)[bs,h_attn,16,q_head]
    输出: A[bs,h_attn,kv_lora] (absorb 结果), V_cache 也写(同 K_cache)
    中间全局 buffer: X16[bs,16,hidden](rmsnorm 输出), Q_out[bs,h_attn,16,q_head] 需预分配传入
    """
    accum = T.float32
    Q_OUT = h_attn * q_head
    half = qk_rope // 2
    q_head_blocks = q_head // 64
    N_QKV = T.ceildiv(Q_OUT, 64)          # 144
    N_KVA = T.ceildiv(kva_out, 64)        # 9
    N_ABS = h_attn * T.ceildiv(kv_lora, 64)  # 128

    @T.prim_func
    def main(
        H_in: T.Tensor([bs, hidden], dtype),
        InLnW: T.Tensor([hidden], dtype),
        QW: T.Tensor([Q_OUT, hidden], dtype),
        QB: T.Tensor([Q_OUT], dtype),
        Cos: T.Tensor([bs, qk_rope], dtype),
        Sin: T.Tensor([bs, qk_rope], dtype),
        KvaW: T.Tensor([kva_out, hidden], dtype),
        KvaB: T.Tensor([kva_out], dtype),
        KvbKn: T.Tensor([h_attn, kv_lora, qk_nope], dtype),
        AbsIdx: T.Tensor([bs * h_attn], T.int32),
        X16: T.Tensor([bs, 16, hidden], dtype),       # 中间: rmsnorm 输出
        QOut: T.Tensor([bs, h_attn, 16, q_head], dtype),  # 中间: pre_qkv 输出
        AOut: T.Tensor([bs * h_attn, kv_lora], dtype),   # 输出: absorb
        BlockTable: T.Tensor([bs, max_seq_blocks], T.int32),
        NewPos: T.Tensor([bs], T.int32),
        Kcache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
        Vcache: T.Tensor([n_blocks, block_size, 1, kva_out], dtype),
    ):
        with T.Kernel(NUM_SMS, threads=128) as (sm_idx,):
            # ===== phase0: rmsnorm (1 task, SM0) =====
            if sm_idx == 0:
                # 单 block rmsnorm: reduce_sq → rsqrt → normalize → 写 X16[0,:]
                sq_frag = T.alloc_fragment([hidden], accum)
                T.clear(sq_frag)
                for i in T.serial(hidden):
                    v = T.cast(H_in[0, i], accum)
                    sq_frag[0] += v * v   # 累加 (单线程串行, hidden=2048 可接受)
                # rsqrt
                rrms = T.alloc_fragment([1], accum)
                rrms[0] = 1.0 / T.sqrt(sq_frag[0] / hidden + 1e-6)
                for i in T.serial(hidden):
                    X16[0, 0, i] = T.cast(T.cast(H_in[0, i], accum) * rrms[0] * T.cast(InLnW[i], accum), dtype)
                    # 其余行清零(M=16 pad)
                for r in T.serial(1, 16):
                    for i in T.serial(hidden):
                        X16[0, r, i] = T.cast(0, dtype)
            T.sync_grid()
            # ===== phase1: pre_qkv(144) ∥ pre_kva(9) =====
            N_TOTAL = N_QKV + N_KVA  # 153
            for task in T.serial(sm_idx, N_TOTAL, NUM_SMS):
                if task < N_QKV:
                    # --- pre_qkv: task = nblk ---
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
                    for j in T.Parallel(qk_rope):
                        cs[j] = Cos[0, j]
                        ss[j] = Sin[0, j]
                    if is_pe:
                        for k in T.Parallel(half):
                            pa = T.cast(acc_s[0, 2 * k], accum) + T.cast(QB[nblk * 64 + 2 * k], accum)
                            pb = T.cast(acc_s[0, 2 * k + 1], accum) + T.cast(QB[nblk * 64 + 2 * k + 1], accum)
                            ck = T.cast(cs[k], accum); sk = T.cast(ss[k], accum)
                            c0 = nblk * 64 + k; c1 = nblk * 64 + k + half
                            h0 = c0 // q_head; d0 = c0 % q_head
                            h1 = c1 // q_head; d1 = c1 % q_head
                            QOut[0, h0, 0, d0] = T.cast(pa * ck - pb * sk, dtype)
                            QOut[0, h1, 0, d1] = T.cast(pa * sk + pb * ck, dtype)
                    else:
                        for j in T.Parallel(64):
                            val = T.cast(acc_s[0, j], accum) + T.cast(QB[nblk * 64 + j], accum)
                            c = nblk * 64 + j; h = c // q_head; d = c % q_head
                            QOut[0, h, 0, d] = T.cast(val, dtype)
                else:
                    # --- pre_kva: task - N_QKV = nblk ---
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


def test():
    """smoke-test：验证本原型 persistent kernel 能跑通 + 输出形状正确。

    早期版本用旧 3-kernel 作独立参考对比数值，但旧 3-kernel 已删除（persistent 已是
    唯一实现且经 e2e token 一致性验证，见 cmp_premla_persist.py）。这里只做 smoke-test。
    """
    torch.manual_seed(42)
    bs=1; H=2048; nh=16; qh=576; qkr=64; qkn=512; kvl=512; kva_out=576
    block_size=16; max_seq_blocks=64; n_blocks=512
    dtype=torch.bfloat16; tl_dt=_TORCH_TO_TL[dtype]; dev="cuda"

    h_in = torch.randn(bs, H, device=dev, dtype=dtype)*0.1
    in_ln_w = torch.ones(H, device=dev, dtype=dtype)
    qw = torch.randn(nh*qh, H, device=dev, dtype=dtype)*0.02
    qb = torch.zeros(nh*qh, device=dev, dtype=dtype)
    cos = torch.randn(bs, qkr, device=dev, dtype=dtype)
    sin = torch.randn(bs, qkr, device=dev, dtype=dtype)
    kvaw = torch.randn(kva_out, H, device=dev, dtype=dtype)*0.02
    kvab = torch.zeros(kva_out, device=dev, dtype=dtype)
    kvb_kn = torch.randn(nh, kvl, qkn, device=dev, dtype=dtype)*0.02
    abs_idx = torch.arange(bs*nh, device=dev, dtype=torch.int32)
    bt = torch.zeros(bs, max_seq_blocks, device=dev, dtype=torch.int32)
    bt[:, 0] = 0
    new_pos = torch.zeros(bs, device=dev, dtype=torch.int32)
    k_cache_p = torch.zeros(n_blocks, block_size, 1, kva_out, device=dev, dtype=dtype)
    v_cache_p = torch.zeros(n_blocks, block_size, 1, kva_out, device=dev, dtype=dtype)

    x16_p = torch.zeros(bs, 16, H, device=dev, dtype=dtype)
    q_out_p = torch.zeros(bs, nh, 16, qh, device=dev, dtype=dtype)
    ker = premla_persistent_kernel(bs, H, nh, qh, qkr, qkn, kvl, kva_out, block_size, max_seq_blocks, n_blocks, tl_dt)
    A_p = ker(h_in, in_ln_w, qw, qb, cos, sin, kvaw, kvab, kvb_kn, abs_idx,
        x16_p, q_out_p, bt, new_pos, k_cache_p, v_cache_p)

    print("=== smoke-test ===")
    print(f"A (absorb) shape: {tuple(A_p.shape)}  expect ({bs*nh}, {kvl})")
    print(f"x16 row0 norm:    {x16_p[0,0,:].float().norm().item():.4f}  (rmsnorm 产物，非零)")
    print(f"k_cache[0,0,0,:] norm: {k_cache_p[0,0,0,:].float().norm().item():.4f}  (store 产物)")
    assert A_p.shape == (bs*nh, kvl), f"A shape mismatch: {A_p.shape}"
    assert x16_p[0,0,:].float().norm().item() > 0, "x16 row0 全零，rmsnorm 未生效"
    print("OK")

    print("\n=== isolation 性能(参考) ===")
    def t(fn, iters=300):
        for _ in range(30): fn()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record();torch.cuda.synchronize()
        return s.elapsed_time(e)/iters*1000
    def pers_path():
        ker(h_in, in_ln_w, qw, qb, cos, sin, kvaw, kvab, kvb_kn, abs_idx,
            x16_p, q_out_p, bt, new_pos, k_cache_p, v_cache_p)
    print(f"persistent(含rmsnorm): {t(pers_path):.1f} us")


if __name__ == "__main__":
    test()
