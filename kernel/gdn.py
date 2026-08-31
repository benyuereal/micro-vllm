"""Qwen3.5 GDN（Gated DeltaNet）Triton kernels。

从 models/qwen3_5/adapter.py 迁出（纯 kernel 代码，归类到 kernel/）。
pointwise/递归类允许 Triton；GEMM 走 gemv/cuBLAS（adapter 侧）。
full-attention 辅助 kernel（qk_norm_rope_partial/rope_partial/attn_gate）
在 kernel/rotary.py（与 Qwen3 的 qk_norm_rope_inplace 同类）。

公开 API：
- gdn_gbeta：g = -exp(A_log)*softplus(a+dt_bias)（fp32），beta = sigmoid(b)
- _gdn_conv_prefill/decode_kernel、_gdn_recurrent_prefill/decode_kernel、
  _gdn_norm_gated_kernel：GDN 层内部 kernel（adapter 直接 launch）
"""
import triton
import triton.language as tl

# =====================================================================

# ---- g = -exp(A_log)*softplus(a+dt_bias)（fp32），beta = sigmoid(b) ----
# A/B 是融合输入投影 buffer 的视图（row stride = STRIDE，非 N）。
@triton.jit
def _gdn_gbeta_kernel(A, B, A_LOG, DT_BIAS, G, BETA, N, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * STRIDE + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + row * STRIDE + cols, mask=mask, other=0.0).to(tl.float32)
    a_log = tl.load(A_LOG + cols, mask=mask, other=0.0).to(tl.float32)
    dt = tl.load(DT_BIAS + cols, mask=mask, other=0.0).to(tl.float32)
    sp = tl.where(a + dt <= 20.0, tl.log(1.0 + tl.exp(a + dt)), a + dt)
    g = -tl.exp(a_log) * sp
    beta = tl.sigmoid(b)
    tl.store(G + row * N + cols, g.to(G.dtype.element_ty), mask=mask)
    tl.store(BETA + row * N + cols, beta.to(BETA.dtype.element_ty), mask=mask)


def gdn_gbeta(a, b, a_log, dt_bias, g_buf, beta_buf, stride):
    M, N = a.shape
    _gdn_gbeta_kernel[(M,)](a, b, a_log, dt_bias, g_buf, beta_buf, N, stride,
                            BLOCK=triton.next_power_of_2(N))
    return g_buf, beta_buf


# ---- GDN short conv1d（causal, kernel 4, groups, silu）+ per-seq conv state ----
# state [POOL, n_gdn, 3, conv_dim] bf16：每 seq 最近 3 个 mixed_qkv（pre-act 输入）。
# 输出 y = x0*w0 + x1*w1 + x2*w2 + x3*w3（x3=当前），silu(y)。
# prefill：处理整段 seq，仅对 seq 末尾 (K-1) 个 token 滚动更新 state。
# decode：单 token，state 滚动更新。
@triton.jit
def _gdn_conv_prefill_kernel(QKV, W, STATE, CU, SEQ_IDX,
                             CONV_DIM, STRIDE, N_GDN, GDN_L, K: tl.constexpr, BLOCK_C: tl.constexpr,
                             CHECKPOINT, CP_N_GDN, CP_ENABLED: tl.constexpr,
                             INIT_STATE, INIT_IDX, INIT_FROM_CP: tl.constexpr):
    pid_c = tl.program_id(0)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < CONV_DIM
    w0 = tl.load(W + c * K + 0, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(W + c * K + 1, mask=cmask, other=0.0).to(tl.float32)
    w2 = tl.load(W + c * K + 2, mask=cmask, other=0.0).to(tl.float32)
    w3 = tl.load(W + c * K + 3, mask=cmask, other=0.0).to(tl.float32)
    s = tl.program_id(1)
    start = tl.load(CU + s)
    end = tl.load(CU + s + 1)
    L = end - start
    sid = tl.load(SEQ_IDX + s)
    # 状态池 [pool, n_gdn, 3, conv_dim]：offset = (slot*n_gdn + 本层 gdn 索引) * 3*conv_dim
    st_base = STATE + (sid.to(tl.int64) * N_GDN + GDN_L) * 3 * CONV_DIM + c
    # 初始 conv 状态：spec 去 rollback 后，非首步 verify 从 checkpoint[accepted_prev]
    # （INIT_STATE 指向 [n_gdn, 3, conv_dim]，token 索引已 bake 进 base 指针）读。
    # conv 是 bf16、recurrent 是 fp32，dtype/布局不同 → 两个 kernel 各自处理。
    if INIT_FROM_CP:
        # CUDA graph 安全：token 索引从 device buffer 读（INIT_IDX[0]），INIT_STATE 指向
        # 完整 checkpoint buffer base [M, n_gdn, 3, conv_dim]（非 bake 进指针的切片视图）。
        init_t = tl.load(INIT_IDX).to(tl.int64)
        st_init = INIT_STATE + (init_t * CP_N_GDN + GDN_L) * 3 * CONV_DIM + c
    else:
        st_init = st_base
    x0 = tl.load(st_init + 0, mask=cmask, other=0.0).to(tl.float32)
    x1 = tl.load(st_init + CONV_DIM, mask=cmask, other=0.0).to(tl.float32)
    x2 = tl.load(st_init + 2 * CONV_DIM, mask=cmask, other=0.0).to(tl.float32)
    for i in range(0, L):
        t = start + i
        x = tl.load(QKV + t.to(tl.int64) * STRIDE + c, mask=cmask, other=0.0).to(tl.float32)
        y = x0 * w0 + x1 * w1 + x2 * w2 + x * w3
        y = y * tl.sigmoid(y)
        tl.store(QKV + t.to(tl.int64) * STRIDE + c, y.to(QKV.dtype.element_ty), mask=cmask)
        # 状态存「含当前 token 的最近 3 个 pre-act 输入」= (x1, x2, x)，与 decode kernel
        # 一致（decode 存 (x1,x2,x)）。若存 (x0,x1,x2) 会漏掉最后一个 prefill token，
        # 首个 decode token 的 conv 输出错位一格。
        if i >= L - (K - 1):
            tl.store(st_base + 0, x1, mask=cmask)
            tl.store(st_base + CONV_DIM, x2, mask=cmask)
            tl.store(st_base + 2 * CONV_DIM, x, mask=cmask)
        # 投机解码：每 token 存 conv 状态检查点（[t, gdn_l, 3, conv_dim]），供接受后回滚。
        if CP_ENABLED:
            cp_base = CHECKPOINT + (t.to(tl.int64) * CP_N_GDN + GDN_L) * 3 * CONV_DIM + c
            tl.store(cp_base + 0, x1.to(CHECKPOINT.dtype.element_ty), mask=cmask)
            tl.store(cp_base + CONV_DIM, x2.to(CHECKPOINT.dtype.element_ty), mask=cmask)
            tl.store(cp_base + 2 * CONV_DIM, x.to(CHECKPOINT.dtype.element_ty), mask=cmask)
        x0 = x1
        x1 = x2
        x2 = x


@triton.jit
def _gdn_conv_decode_kernel(QKV, W, STATE, SEQ_IDX, IS_REAL,
                            CONV_DIM, STRIDE, N_GDN, GDN_L, K: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    # pad 行（循环复制）：不更新状态、不算输出。IS_REAL 是 buffer（graph 安全：
    # replay 时重读，非 capture 时 bake 的标量）。
    if tl.load(IS_REAL + row) == 0:
        return
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < CONV_DIM
    sid = tl.load(SEQ_IDX + row)
    # 状态池 [pool, n_gdn, 3, conv_dim]：offset = (slot*n_gdn + 本层 gdn 索引) * 3*conv_dim
    st_base = STATE + (sid.to(tl.int64) * N_GDN + GDN_L) * 3 * CONV_DIM + c
    x0 = tl.load(st_base + 0, mask=cmask, other=0.0).to(tl.float32)
    x1 = tl.load(st_base + CONV_DIM, mask=cmask, other=0.0).to(tl.float32)
    x2 = tl.load(st_base + 2 * CONV_DIM, mask=cmask, other=0.0).to(tl.float32)
    w0 = tl.load(W + c * K + 0, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(W + c * K + 1, mask=cmask, other=0.0).to(tl.float32)
    w2 = tl.load(W + c * K + 2, mask=cmask, other=0.0).to(tl.float32)
    w3 = tl.load(W + c * K + 3, mask=cmask, other=0.0).to(tl.float32)
    x = tl.load(QKV + row.to(tl.int64) * STRIDE + c, mask=cmask, other=0.0).to(tl.float32)
    y = x0 * w0 + x1 * w1 + x2 * w2 + x * w3
    y = y * tl.sigmoid(y)
    tl.store(QKV + row.to(tl.int64) * STRIDE + c, y.to(QKV.dtype.element_ty), mask=cmask)
    tl.store(st_base + 0, x1, mask=cmask)
    tl.store(st_base + CONV_DIM, x2, mask=cmask)
    tl.store(st_base + 2 * CONV_DIM, x, mask=cmask)


# ---- GDN 递归状态更新（delta rule，fp32 state）----
# state_pool [POOL, n_gdn, H, DK, DV] fp32。
# qkv 布局 [M, 2*H*DK + H*DV]：q[0:H*DK] k[H*DK:2*H*DK] v[2*H*DK:]。
# decode：每 (row, head) 一个 program；prefill：每 (seq, head) 一个 program 循环 token。

@triton.jit
def _gdn_delta_step(S_m, q, k, v, g, beta, SCALE: tl.constexpr):
    """单步 delta rule（decode/prefill 共享，Triton 内联）：
    l2norm(q,k) → S*=exp(g) → kv_mem=S@k → delta=(v-kv_mem)*beta → S+=k⊗delta → o=S@q。
    返回 (S_m, o)。q/k/v 以 fp32 参与（对齐 HF：l2norm/scale/累加全 fp32）。
    注意：kv_mem[j] = sum_i S[i,j]*k[i]，k 须沿 DK 轴（axis 0）广播 → k[:, None]。
    误用 k[None, :] 会沿 DV 轴广播，算成 k[j]*sum_i S[i,j]（方向错，state 全错）。
    """
    q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * SCALE
    k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    S_m = S_m * tl.exp(g)
    kv_mem = tl.sum(S_m * k[:, None], axis=0)
    delta = (v - kv_mem) * beta
    S_m += k[:, None] * delta[None, :]
    o = tl.sum(S_m * q[:, None], axis=0)
    return S_m, o


@triton.jit
def _gdn_recurrent_decode_kernel(QKV, G, BETA, STATE, OUT, SEQ_IDX, IS_REAL,
                                 H: tl.constexpr, HK: tl.constexpr,
                                 DK: tl.constexpr, DV: tl.constexpr,
                                 N_GDN, GDN_L, SCALE, STRIDE, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    if tl.load(IS_REAL + row) == 0:
        return  # pad 行：不更新状态、不算输出（graph 安全：IS_REAL 是 buffer）
    sid = tl.load(SEQ_IDX + row)
    # 状态池 [pool, n_gdn, H, DK, DV]：offset = (slot*n_gdn + 本层 gdn 索引) * H*DK*DV
    S = STATE + (sid.to(tl.int64) * N_GDN + GDN_L) * H * DK * DV + h * DK * DV
    dk = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_D)
    S_m = tl.load(S + dk[:, None] * DV + dv[None, :]).to(tl.float32)

    # q/k 只有 HK 个 head（27B: HK=16, H=48），HF 用 repeat_interleave(H//HK) 扩到 H：
    # 递归 head h 的 q/k = 原始 q/k head (h // (H//HK))。q/k 段宽 HK*DK（非 H*DK）。
    # v 有 H 个 head（value_dim = H*DV），v head h 在 2*HK*DK + h*DV。
    hk = h // (H // HK)
    q_base = QKV + row.to(tl.int64) * STRIDE + hk * DK
    k_base = QKV + row.to(tl.int64) * STRIDE + HK * DK + hk * DK
    v_base = QKV + row.to(tl.int64) * STRIDE + 2 * HK * DK + h * DV
    # q/k/v 以 fp32 参与递归（对齐 HF：l2norm/scale/累加全 fp32）。
    q = tl.load(q_base + dk).to(tl.float32)
    k = tl.load(k_base + dk).to(tl.float32)
    v = tl.load(v_base + dv).to(tl.float32)
    g = tl.load(G + row * H + h).to(tl.float32)
    beta = tl.load(BETA + row * H + h).to(tl.float32)

    S_m, o = _gdn_delta_step(S_m, q, k, v, g, beta, SCALE)

    tl.store(S + dk[:, None] * DV + dv[None, :], S_m.to(STATE.dtype.element_ty))
    o_base = OUT + row.to(tl.int64) * (H * DV) + h * DV
    tl.store(o_base + dv, o.to(OUT.dtype.element_ty))


@triton.jit
def _gdn_recurrent_prefill_kernel(QKV, G, BETA, STATE, OUT, CU, SEQ_IDX,
                                  H: tl.constexpr, HK: tl.constexpr,
                                  DK: tl.constexpr, DV: tl.constexpr,
                                  N_GDN, GDN_L, SCALE, STRIDE, BLOCK_D: tl.constexpr,
                                  CHECKPOINT, CP_N_GDN, CP_ENABLED: tl.constexpr,
                                  INIT_STATE, INIT_IDX, INIT_FROM_CP: tl.constexpr):
    s = tl.program_id(0)
    h = tl.program_id(1)
    start = tl.load(CU + s)
    end = tl.load(CU + s + 1)
    L = end - start
    sid = tl.load(SEQ_IDX + s)
    # 状态池 [pool, n_gdn, H, DK, DV]：offset = (slot*n_gdn + 本层 gdn 索引) * H*DK*DV
    S = STATE + (sid.to(tl.int64) * N_GDN + GDN_L) * H * DK * DV + h * DK * DV
    dk = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_D)
    # 初始状态：spec 去 rollback 后，非首步 verify 直接从 checkpoint[accepted_prev]
    # （INIT_STATE 指向 [n_gdn, H, DK, DV]，token 索引已 bake 进 base 指针）读，
    # 省掉接受后 copy_ 回 pool 的 DtoD。首步 verify / 正常 prefill 仍从 pool 读。
    # 初始 load 在循环前，checkpoint store 在循环内 → 无读写竞争。
    if INIT_FROM_CP:
        # CUDA graph 安全：token 索引从 device buffer 读（INIT_IDX[0]），INIT_STATE 指向
        # 完整 checkpoint buffer base [M, n_gdn, H, DK, DV]（非 bake 进指针的切片视图）。
        init_t = tl.load(INIT_IDX).to(tl.int64)
        S_init = INIT_STATE + (init_t * CP_N_GDN + GDN_L) * H * DK * DV + h * DK * DV
    else:
        S_init = S
    S_m = tl.load(S_init + dk[:, None] * DV + dv[None, :]).to(tl.float32)
    # q/k 只有 HK 个 head，HF repeat_interleave(H//HK) 扩到 H（见 decode kernel 注释）。
    hk = h // (H // HK)

    for i in range(0, L):
        t = start + i
        q_base = QKV + t.to(tl.int64) * STRIDE + hk * DK
        k_base = QKV + t.to(tl.int64) * STRIDE + HK * DK + hk * DK
        v_base = QKV + t.to(tl.int64) * STRIDE + 2 * HK * DK + h * DV
        q = tl.load(q_base + dk).to(tl.float32)
        k = tl.load(k_base + dk).to(tl.float32)
        v = tl.load(v_base + dv).to(tl.float32)
        g = tl.load(G + t.to(tl.int64) * H + h).to(tl.float32)
        beta = tl.load(BETA + t.to(tl.int64) * H + h).to(tl.float32)
        S_m, o = _gdn_delta_step(S_m, q, k, v, g, beta, SCALE)
        o_base = OUT + t.to(tl.int64) * (H * DV) + h * DV
        tl.store(o_base + dv, o.to(OUT.dtype.element_ty))
        # 投机解码：每 token 存递归状态检查点（[t, gdn_l, H, DK, DV]），供接受后回滚。
        if CP_ENABLED:
            cp_base = CHECKPOINT + (t.to(tl.int64) * CP_N_GDN + GDN_L) * H * DK * DV + h * DK * DV
            tl.store(cp_base + dk[:, None] * DV + dv[None, :], S_m.to(CHECKPOINT.dtype.element_ty))

    tl.store(S + dk[:, None] * DV + dv[None, :], S_m.to(STATE.dtype.element_ty))


# ---- GDN RMSNormGated：out = (o * rrms * w) * silu(z)，per (row, head) on DV ----
# 注意：HF Qwen3_5RMSNormGated 用 self.weight * hidden（非 1-centered）。
@triton.jit
def _gdn_norm_gated_kernel(O, Z, W, OUT, H: tl.constexpr, DV: tl.constexpr, eps,
                           Z_STRIDE, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    dv = tl.arange(0, BLOCK_D)
    o = tl.load(O + row.to(tl.int64) * (H * DV) + h * DV + dv).to(tl.float32)
    z = tl.load(Z + row.to(tl.int64) * Z_STRIDE + h * DV + dv).to(tl.float32)
    w = tl.load(W + dv).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(o * o) / DV + eps)
    y = o * rrms * w * (z * tl.sigmoid(z))
    tl.store(OUT + row.to(tl.int64) * (H * DV) + h * DV + dv, y.to(OUT.dtype.element_ty))
