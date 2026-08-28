"""Qwen3_5Adapter - Qwen3.5 (GDN 线性注意力 + full attention 混合) 适配器。

正确性基准：HF transformers 5.15.1 models/qwen3_5/modeling_qwen3_5.py。

架构要点：
- 24 层混合：layer_types 里 "linear_attention"(GDN) 与 "full_attention" 交替
  （full_attention_interval=4：第 4/8/12/16/20/24 层 full）。
- RMSNorm 是 1-centered：out = x * rrms * (1 + w)（Qwen3 是 x * w）。
  用 kernel.rmsnorm 的 rmsnorm1* 系列。
- full attention：q_proj 输出 2*heads*head_dim，view(-1, head_dim*2) 后 chunk(2)
  → 交错布局：query = 偶数 head_dim 块、gate = 奇数块（非连续！）。
  QK-Norm（1-centered, per-head on head_dim），partial rotary（前 64 维 half-split），
  attn_output_gate：attn * sigmoid(gate)，o_proj。head_dim=256, 8q/2kv。
- GDN（Gated DeltaNet）：
  * in_proj_qkv [6144,1024] → q(2048)|k(2048)|v(2048)，16 kv heads × 128
  * short conv1d（kernel 4, groups, silu）作用在 qkv，per-seq conv state（存 pre-act 输入）
  * beta = sigmoid(b)，g = -exp(A_log)*softplus(a+dt_bias)（fp32）
  * q/k l2norm(eps 1e-6)；q 额外乘 scale = 1/sqrt(DK)
  * 递归状态 S [16,128,128] fp32（delta rule）：
      S = S*exp(g); delta = (v - S@k)*beta; S += outer(k, delta); o = S@q
  * out = RMSNormGated(o, z) * silu(z)（norm 非 1-centered，权重直接乘），再 out_proj
- GDN 状态（recurrent + conv）存 per-seq 池（state_pool[seq_id % POOL]），
  decode 仅真实 seq 行（row < n_real）更新，pad 行（循环复制）跳过。
- RoPE：theta 1e7，partial 0.25（rope_dim=64），mrope 纯文本退化为 1D position。
  复用 PagedAttention 的 _cos_pool/_sin_pool（[max_pos, 32]，inv_freq 前 32 个）。
- tie_word_embeddings=true：lm_head = embed_tokens。
- 多模态壳：AutoModelForCausalLM → Qwen3_5ForCausalLM（纯文本），model.config 即
  text_config，model.model 即 text model（embed_tokens/layers/norm），vision 不加载。
"""
import os
import torch
import triton
import triton.language as tl

from models.base import ModelAdapter
from kernel.rmsnorm import (
    rmsnorm1, rmsnorm1_, rmsnorm1_residual_gemm as rmsnorm1_residual,
    rmsnorm1_residual_fused,
)
from kernel.dense_mlp import dense_swiglu
from kernel.gemv import gemv_or_matmul, gemv_v2
from kernel.gemv_int8 import w8_linear
from kernel.quant import quantize_per_channel
from kernel import marlin as _marlin
from core.cache_manager import store_kvcache

# W8A16 开关：MICRO_W8A16=1 时权重 INT8（per-channel）+ 激活 bf16。
# decode（memory-bound）权重带宽减半；prefill（compute-bound）反量化后 matmul。
_W8A16 = os.environ.get("MICRO_W8A16", "0") == "1"

# verify int8 GEMM 后端：MICRO_VERIFY_GEMM=marlin|tilelang，默认 marlin。
# marlin 模式：权重存 Marlin 格式（repacked int8，同字节数），verify/decode/prefill
#   的 int8 线性全走 marlin_forward（CUTLASS C++，int8 tensor-core mma）。
# tilelang 模式：保持 (int8 [N,K], scale) 元组 + TileLang/Triton GEMM（原行为）。
# 显存约束：Marlin repacked 与 int8 [N,K] 同字节（24.33GB），不能两者共存（~67G OOM），
#   故 marlin 模式用 Marlin 格式【替换】int8（非额外保留）。
_MICRO_VERIFY_GEMM = os.environ.get("MICRO_VERIFY_GEMM", "marlin").lower()
_MARLIN = _MICRO_VERIFY_GEMM == "marlin"

# lm_head int8（Marlin）开关：MICRO_LMHEAD_INT8=1 时把 bf16 lm_head（2.54GB，quant
# ignore 列表存 bf16）group-128 量化成 int8 Marlin（1.27GB），forward 走 marlin_forward。
# 收益：lm_head 每 spec step 被调 2 次（verify M=8 + draft M=7 共享 target lm_head），
# bf16 每次读 2.54GB（3.53ms）→ int8 读 1.27GB（1.77ms）→ 省 ~3.5ms/step（~6.5%）。
# 正确性：lm_head 被 verify/draft/非spec decode 共享，全路径对称走 int8 → spec==非spec
# 等价性保持（两者一致），draft/target 一致（接受率保持）。模型本身已 W8A16（64 层
# int8），lm_head 在 quant ignore 列表存 bf16——转 int8 是【补全】W8A16 量化，非新近似。
# 默认开（MICRO_LMHEAD_INT8=0 可关）：per_step 55.61→50.65ms(-6.3%)，acceptance 保持。
_LMHEAD_INT8 = os.environ.get("MICRO_LMHEAD_INT8", "1") == "1"

try:
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
except ImportError:
    flash_attn_with_kvcache = None
    flash_attn_varlen_func = None


# =====================================================================
# GDN Triton kernels（pointwise/递归类，允许 Triton；GEMM 走 gemv/cuBLAS）
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

    q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * SCALE
    k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)

    ge = tl.exp(g)
    S_m = S_m * ge
    # kv_mem[j] = sum_i S[i,j]*k[i]：k 须沿 DK 轴（axis 0）广播 → k[:, None]。
    # 误用 k[None, :] 会沿 DV 轴广播，算成 k[j]*sum_i S[i,j]（方向错，state 全错）。
    kv_mem = tl.sum(S_m * k[:, None], axis=0)
    delta = (v - kv_mem) * beta
    S_m += k[:, None] * delta[None, :]
    o = tl.sum(S_m * q[:, None], axis=0)

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
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        ge = tl.exp(g)
        S_m = S_m * ge
        # kv_mem[j] = sum_i S[i,j]*k[i]：k 沿 DK 轴广播 → k[:, None]（见 decode kernel 注释）
        kv_mem = tl.sum(S_m * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        S_m += k[:, None] * delta[None, :]
        o = tl.sum(S_m * q[:, None], axis=0)
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


# ---- full attention decode：QK-Norm(1-centered) + partial RoPE（前 rot 维 half-split）----
@triton.jit
def _qk_norm_rope_partial_kernel(QKV, W, COS, SIN, POS,
                                 stride_qkv_row, seg_offset, head_size: tl.constexpr,
                                 num_heads: tl.constexpr, rot: tl.constexpr,
                                 eps, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    base = batch_idx * stride_qkv_row + seg_offset + head_idx * head_size
    pos = tl.load(POS + batch_idx)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < head_size
    x = tl.load(QKV + base + offs, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=0) / head_size + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    xn = x * rrms * (1.0 + w)

    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < rot // 2
    c = tl.load(COS + pos * (rot // 2) + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    s = tl.load(SIN + pos * (rot // 2) + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    x1 = tl.load(QKV + base + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    x2 = tl.load(QKV + base + rot // 2 + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W + rot // 2 + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    xn1 = x1 * rrms * (1.0 + w1)
    xn2 = x2 * rrms * (1.0 + w2)
    o1 = (xn1 * c - xn2 * s).to(QKV.dtype.element_ty)
    o2 = (xn2 * c + xn1 * s).to(QKV.dtype.element_ty)
    tl.store(QKV + base + r_offs, o1, mask=r_mask)
    tl.store(QKV + base + rot // 2 + r_offs, o2, mask=r_mask)
    p_mask = (offs >= rot) & (offs < head_size)
    tl.store(QKV + base + offs, xn.to(QKV.dtype.element_ty), mask=p_mask)


def qk_norm_rope_partial_inplace(qkv_buf, bs, seg_offset, num_heads, head_size,
                                 norm_weight, cos_pool, sin_pool, positions, eps=1e-6):
    rot = cos_pool.shape[1] * 2
    BLOCK_H = triton.next_power_of_2(head_size)
    BLOCK_R = triton.next_power_of_2(rot // 2)
    _qk_norm_rope_partial_kernel[(bs * num_heads,)](
        qkv_buf, norm_weight, cos_pool, sin_pool, positions,
        qkv_buf.stride(0), seg_offset, head_size, num_heads, rot,
        eps, BLOCK_H=BLOCK_H, BLOCK_R=BLOCK_R)


# ---- prefill 纯 partial RoPE（无 norm）：in-place half-split 前 rot 维 ----
# 替代 _prefill_full 里 PyTorch 的 cos/sin gather + 4 slice + 4 mul + 2 add + 2 cat
# （每 (q,k) 张量 ~12 个小 kernel × 2 × 16 full 层 = ~384 次 launch/verify）。
# 一个 program = (token, head)，只读写前 rot 维（rot 之后维度不动）。
@triton.jit
def _rope_partial_inplace_kernel(X, COS, SIN, POS,
                                 stride_t, stride_h,
                                 head_size: tl.constexpr, rot: tl.constexpr,
                                 BLOCK_R: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(POS + t)
    base = t.to(tl.int64) * stride_t + h * stride_h
    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < rot // 2
    c = tl.load(COS + pos * (rot // 2) + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    s = tl.load(SIN + pos * (rot // 2) + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    x1 = tl.load(X + base + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + rot // 2 + r_offs, mask=r_mask, other=0.0).to(tl.float32)
    o1 = (x1 * c - x2 * s).to(X.dtype.element_ty)
    o2 = (x2 * c + x1 * s).to(X.dtype.element_ty)
    tl.store(X + base + r_offs, o1, mask=r_mask)
    tl.store(X + base + rot // 2 + r_offs, o2, mask=r_mask)


def rope_partial_inplace(x, cos_pool, sin_pool, positions):
    """x [T, H, head_dim] in-place partial RoPE（前 rot 维 half-split，rot 后不动）。
    positions [T] int64。cos/sin 表 [max_pos, rot//2]（PagedAttention 池）。"""
    T, H, hd = x.shape
    rot = cos_pool.shape[1] * 2
    BLOCK_R = triton.next_power_of_2(rot // 2)
    _rope_partial_inplace_kernel[(T, H)](
        x, cos_pool, sin_pool, positions,
        x.stride(0), x.stride(1),
        head_size=hd, rot=rot, BLOCK_R=BLOCK_R)


# ---- attn_output_gate：out = attn * sigmoid(gate)（fp32 sigmoid，bf16 输出）----
# 替代 PyTorch 的 sigmoid + cast + mul 三个 elementwise kernel（× 16 full 层）。
@triton.jit
def _attn_gate_kernel(ATTN, GATE, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(ATTN + offs).to(tl.float32)
    g = tl.load(GATE + offs).to(tl.float32)
    tl.store(ATTN + offs, (a * tl.sigmoid(g)).to(ATTN.dtype.element_ty))


def attn_gate_inplace(attn, gate):
    """attn [T, nh, hd]（in-place *= sigmoid(gate)），gate 同形状。"""
    n = attn.numel()
    BLOCK = 1024
    _attn_gate_kernel[(triton.cdiv(n, BLOCK),)](attn, gate, BLOCK=BLOCK)


# =====================================================================
# Adapter
# =====================================================================

class Qwen3_5Adapter(ModelAdapter):
    model_type = "qwen3_5"

    # full attention 层走 prerope+store+pure-flash（slot_mapping/flash_seqlens 由
    # model_graph 在 use_prerope_decode=True 时统一计算）。
    use_prerope_decode = True

    def __init__(self):
        self._tcfg = None
        self._layer_types = None
        self._n_gdn = 0
        self._n_full = 0
        self._gdn_H = 16
        self._gdn_HK = 16
        self._gdn_DK = 128
        self._gdn_DV = 128
        self._gdn_conv_dim = 6144
        self._gdn_K = 4
        self._rot = 64

    # -------------------- 模块访问（多模态壳） --------------------
    # Qwen3_5ForConditionalGeneration：model.model = Qwen3_5Model，其 .language_model
    # 才是文本模型（embed_tokens/layers/norm）。base 默认 model.model.* 会拿到壳，
    # 故 override 到 model.model.language_model.*。lm_head 在壳顶层（model.lm_head）。
    def _text_model(self, model):
        m = model.model
        return getattr(m, "language_model", m)

    def embed(self, model):
        return self._text_model(model).embed_tokens

    def blocks(self, model):
        return self._text_model(model).layers

    def final_norm(self, model):
        return self._text_model(model).norm

    def final_norm_one_centered(self) -> bool:
        # Qwen3_5RMSNorm 是 1-centered：out = x*rrms*(1+w)（非 Qwen3 的 x*w）。
        return True

    def lm_head(self, model):
        return model.lm_head

    # -------------------- 元信息 --------------------
    def _tc(self, cfg):
        if self._tcfg is None:
            tc = getattr(cfg, "text_config", None)
            self._tcfg = tc if tc is not None else cfg
            self._layer_types = list(self._tcfg.layer_types)
            self._n_gdn = sum(1 for t in self._layer_types if t == "linear_attention")
            self._n_full = sum(1 for t in self._layer_types if t == "full_attention")
            self._gdn_H = self._tcfg.linear_num_value_heads
            self._gdn_HK = self._tcfg.linear_num_key_heads
            self._gdn_DK = self._tcfg.linear_key_head_dim
            self._gdn_DV = self._tcfg.linear_value_head_dim
            self._gdn_conv_dim = (2 * self._tcfg.linear_num_key_heads * self._gdn_DK
                                  + self._gdn_H * self._gdn_DV)
            self._gdn_K = self._tcfg.linear_conv_kernel_dim
            rp = self._tcfg.rope_parameters
            prf = rp.get("partial_rotary_factor", 1.0) if isinstance(rp, dict) else 1.0
            self._rot = int(self._tcfg.head_dim * prf)
        return self._tcfg

    def cache_dims(self, cfg):
        tc = self._tc(cfg)
        return tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim

    def num_layers(self, cfg):
        return self._tc(cfg).num_hidden_layers

    def intermediate_size(self, cfg, world_size):
        return self._tc(cfg).intermediate_size // world_size

    def rope_dim(self, cfg):
        self._tc(cfg)
        return self._rot

    def rope_theta(self, cfg):
        tc = self._tc(cfg)
        rp = tc.rope_parameters
        return (rp.get("rope_theta", 10000.0) if isinstance(rp, dict) else 10000.0)

    def softmax_scale(self, cfg):
        return self._tc(cfg).head_dim ** -0.5

    def supports_chunked_prefill(self, cfg) -> bool:
        return True

    @staticmethod
    def _ln_eps(ln, cfg):
        return getattr(ln, "eps", None) or getattr(ln, "variance_epsilon", cfg.rms_norm_eps)

    # -------------------- W8A16 统一线性分派 --------------------
    def _q(self, w):
        """W8A16 开启时 bf16 [N,K] → (w_int8, scale) 元组；否则原样返回。
        已是 (int8, scale) 元组（预量化模型，如 Qwen3.8）→ 原样返回（不重复量化）。"""
        if isinstance(w, tuple):
            return w
        if _W8A16:
            return quantize_per_channel(w)
        return w

    @staticmethod
    def _is_marlin(w):
        """w 是否为 Marlin 格式 dict（marlin 模式的 W8A16 权重）。"""
        return isinstance(w, dict) and "wq" in w

    def _to_marlin(self, w_int8, scale):
        """(int8 [N,K], scale fp32 [N,K/128]) → Marlin 格式 dict。

        int8 → packed int32 [N,K/4]（byte-128 编码）→ build_marlin（gptq_marlin_repack
        + permute/pad scales）。Marlin repacked 与 int8 同字节数，转换后 int8 可释放
        （不增显存）。scale fp32→bf16（build_marlin 期望 bf16，与 checkpoint 一致）。"""
        assert _marlin.marlin_available(), "MICRO_VERIFY_GEMM=marlin 但 Marlin kernel 不可用"
        N, K = w_int8.shape
        packed = _marlin.int8_to_packed(w_int8)      # int32 [N,K/4]
        scale_bf16 = scale.to(torch.bfloat16)         # [N,K/128]
        m = _marlin.build_marlin(packed, scale_bf16, N, K, w_int8.device)
        del packed, scale_bf16
        return m

    def _store_w(self, w):
        """存权重：marlin 模式且 w 是 group-128 (int8, scale[N,K/128]) 元组 → 转 Marlin
        dict（替换 int8，同字节数不增显存）；否则原样返回（bf16 / tilelang int8 元组 /
        per-channel 自量化 scale[N]——Marlin 只支持 group-128，per-channel 保持原路径）。"""
        if _MARLIN and isinstance(w, tuple) and w[1].dim() == 2:
            return self._to_marlin(w[0], w[1])
        return w

    def _unpack_linear(self, mod, world_size, rank, chunk_dim=0):
        """从 Linear 模块取权重，统一成 bf16 [N,K] 或 (int8 [N,K], scale [N,K/128])。

        - 普通 nn.Linear（.weight bf16）→ 返回 bf16 [N,K]（TP 按 chunk_dim 切）。
        - pack-quantized（weight_packed int32 [N,K/4] + weight_scale bf16 [N,K/128]，
          Qwen3.8 W8A16 预量化）→ 解包成 (int8 [N,K], scale fp32 [N,K/128])。
          解包：每 int32 打包 4 个 int8，byte i（bits 8i..8i+7）= (int8+128)&0xFF
          → int8 = byte - 128（非补码）；stack 沿 dim=2（byte 位置是 minor 维）。
          已验证与 compressed_tensors 参考 dequantize 完全一致（max diff 0.0）。
        """
        if hasattr(mod, "weight_packed"):
            # int32 packed [N,K/4] 与 int8 [N,K] 同字节数。若在 GPU 上解包，int32 未释放
            # 前 int8 已分配 → 双份常驻（27B 54G 超 GPU4 45G OOM）。故先在 CPU 解包
            # （RAM 875G 充足），只把 int8 结果搬上 GPU，int32 在 CPU 释放。
            dev = mod.weight_packed.device
            packed = mod.weight_packed.to("cpu")   # int32 [N, K/4] → CPU
            scale = mod.weight_scale.to("cpu").float()  # bf16 [N, K/128] → CPU fp32
            N, K4 = packed.shape
            K = K4 * 4
            p = packed.to(torch.int32)
            b0 = (p & 0xFF).to(torch.int32)
            b1 = ((p >> 8) & 0xFF).to(torch.int32)
            b2 = ((p >> 16) & 0xFF).to(torch.int32)
            b3 = ((p >> 24) & 0xFF).to(torch.int32)
            w_int8 = torch.stack([b0 - 128, b1 - 128, b2 - 128, b3 - 128],
                                 dim=2).reshape(N, K).to(torch.int8)
            del packed, p, b0, b1, b2, b3  # 释放 CPU int32
            # 释放 GPU 上的 int32 packed + scale（模块属性）
            del mod.weight_packed
            del mod.weight_scale
            if hasattr(mod, "weight_shape"):
                del mod.weight_shape
            w_int8 = w_int8.chunk(world_size, dim=chunk_dim)[rank].to(dev)
            scale = scale.chunk(world_size, dim=chunk_dim)[rank].to(dev)
            return w_int8.contiguous(), scale.contiguous()
        w = mod.weight.data
        return w.chunk(world_size, dim=chunk_dim)[rank].contiguous()

    @staticmethod
    def _reorder_qgate(w, nh, hd):
        """q_proj 权重行重排 [query 全部 | gate 全部]（交错 → 连续）。
        w 为 bf16 [2*nh*hd, K] 或 (int8 [2*nh*hd, K], scale [2*nh*hd, K/128])。
        行重排对两者都成立（scale 随行一起重排）。"""
        if isinstance(w, tuple):
            wq, sc = w
            wq = wq.view(nh, 2 * hd, -1)
            wq = torch.cat([wq[:, :hd, :].reshape(-1, wq.shape[-1]),
                            wq[:, hd:, :].reshape(-1, wq.shape[-1])], dim=0).contiguous()
            sc = sc.view(nh, 2 * hd, -1)
            sc = torch.cat([sc[:, :hd, :].reshape(-1, sc.shape[-1]),
                            sc[:, hd:, :].reshape(-1, sc.shape[-1])], dim=0).contiguous()
            return wq, sc
        w = w.view(nh, 2 * hd, -1)
        return torch.cat([w[:, :hd, :].reshape(-1, w.shape[-1]),
                          w[:, hd:, :].reshape(-1, w.shape[-1])], dim=0).contiguous()

    @staticmethod
    def _cat_w(ws):
        """cat 一组权重（bf16 或 int8 元组，须同 dtype 类型）沿 dim=0。"""
        if isinstance(ws[0], tuple):
            return (torch.cat([w[0] for w in ws], dim=0).contiguous(),
                    torch.cat([w[1] for w in ws], dim=0).contiguous())
        return torch.cat(ws, dim=0).contiguous()

    def _lin(self, x, w, out=None, env="MICRO_GEMV"):
        """统一线性：w 为 bf16 [N,K] / (w_int8, scale) 元组（W8A16 tilelang）/
        Marlin dict（W8A16 marlin 模式）。
        - Marlin dict → marlin_forward（CUTLASS C++ int8 GEMM，全 M 通用）。
        - (int8, scale) → w8_linear（M=1 int8 GEMV / M>1 反量化 matmul）。
        - bf16 → gemv_or_matmul。"""
        if self._is_marlin(w):
            return _marlin.marlin_forward(w, x, out)
        if isinstance(w, tuple):
            return w8_linear(x, w[0], w[1], out, env)
        if out is None:
            out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
        return gemv_or_matmul(x, w, out, env)

    def _lin_prefill(self, x, w):
        """prefill 线性（M=T>1）：w 为 bf16 [N,K] 或 (w_int8, scale) 元组。
        反量化后 x @ w.t()（compute-bound，反量化开销可忽略）。
        scale 两种：[N]（per-channel）→ unsqueeze(1)；[N,K/128]（group-128）→ repeat_interleave 128。

        投机解码 verify（M 小，verify_gemm 开）：group-128 int8 走双后端 int8 GEMM
        （TileLang 默认 / Triton 备选，MICRO_VERIFY_GEMM 切换；权重 HBM 只读一次，
        shared 内 dequant→bf16 + GEMM），避免反量化 54GB bf16 权重/层。

        小 M prefill（M≤128，如投机解码 prompt prefill M≈61）：同样走 int8 GEMM。
        原路径反量化整份 int8 权重到 bf16（w_int8.float()*sc 物化 4x fp32 临时 +
        bf16 拷贝，mlp_gu M=61 反量化占 7.29ms/7.9ms），int8 GEMM 权重 HBM 只读
        一次（0.35ms，快 22x）。M>128 时 int8 GEMM 的 BLOCK_M 超 shared mem 上限
        （M=256 编译失败），回退反量化 matmul（正常大 batch prefill 不受影响）。

        Marlin dict（marlin 模式）：全 M 走 marlin_forward（CUTLASS C++ int8 GEMM，
        权重 HBM 只读一次，M 大时 tensor-core 跨 M 复用权重，比反量化 matmul 快）。"""
        if self._is_marlin(w):
            return _marlin.marlin_forward(w, x)
        if isinstance(w, tuple):
            w_int8, scale = w
            if scale.dim() == 2:
                from kernel.gemm_int8_triton import verify_int8_gemm
                if x.shape[0] <= 128:
                    return verify_int8_gemm(x, w_int8, scale)
                sc = scale.repeat_interleave(128, dim=1)
            else:
                sc = scale.unsqueeze(1)
            w = (w_int8.float() * sc).to(x.dtype)
        return torch.matmul(x, w.t())

    # -------------------- 权重预处理 --------------------
    def prepare_weights(self, model, world_size, rank):
        blocks = self.blocks(model)
        if getattr(blocks[0], "_prepared", False):
            return
        cfg = model.config
        tc = self._tc(cfg)

        gdn_layer_idx = 0  # GDN 层在状态池 n_gdn 维的索引（0..n_gdn-1）
        for block in blocks:
            is_gdn = hasattr(block, "linear_attn") and block.linear_attn is not None
            if is_gdn:
                la = block.linear_attn
                la._gdn_layer_idx = gdn_layer_idx
                gdn_layer_idx += 1
                # 输入投影：qkv/z（可能 int8 预量化）+ b/a（始终 bf16，在 ignore 列表）。
                # 0.8B（bf16）：4 个全 bf16 → 融合成 1 个 [8224, hidden] GEMV（b/a 仅 N=16
                #   单独 GEMV 会 92 SM 空转 + 3 次额外 launch，融合省 ~180us/step）。
                # 27B W8A16（qkv/z int8、b/a bf16）：dtype 不同不能混融 → 2 个 GEMV：
                #   qkv+z 融成 1 个 int8 GEMV [16384, 5120]，b+a 融成 1 个 bf16 GEMV [96, 5120]。
                # 下游 kernel 用 row-stride 读对应 buffer 的段（qkv/z 视图 / b/a 视图）。
                qkv_w = self._unpack_linear(la.in_proj_qkv, world_size, rank)
                z_w = self._unpack_linear(la.in_proj_z, world_size, rank)
                b_w = self._unpack_linear(la.in_proj_b, world_size, rank)
                a_w = self._unpack_linear(la.in_proj_a, world_size, rank)
                la._gdn_w8 = isinstance(qkv_w, tuple)
                if la._gdn_w8:
                    # int8：cat int8 权重 + cat scale（group-128，scale [N, K/128]）。
                    # marlin 模式：_store_w 把 (int8, scale) 转 Marlin dict（替换 int8）。
                    la._in_w_qz = self._store_w(
                        (torch.cat([qkv_w[0], z_w[0]], dim=0).contiguous(),
                         torch.cat([qkv_w[1], z_w[1]], dim=0).contiguous()))
                else:
                    # bf16：4 个全融合成 1 个 GEMV（原 0.8B 路径）
                    la._in_w = torch.cat([qkv_w, z_w, b_w, a_w], dim=0).contiguous()
                la._in_w_ba = torch.cat([b_w, a_w], dim=0).contiguous()  # [2H, hidden] bf16
                la._o_w = self._store_w(self._q(self._unpack_linear(la.out_proj, world_size, rank)))
                la._conv_w = la.conv1d.weight.data.squeeze(1).contiguous()  # [6144, 4]（conv 不量化）
                la._a_log = la.A_log.data.float().contiguous()        # [16] fp32
                la._dt_bias = la.dt_bias.data.float().contiguous()    # [16] fp32
                la._norm_w = la.norm.weight.data.float().contiguous()  # [128] fp32
                la._norm_eps = self._ln_eps(la.norm, cfg)
                la.in_proj_qkv = la.in_proj_z = la.in_proj_b = la.in_proj_a = None
                la.out_proj = la.conv1d = la.A_log = la.dt_bias = la.norm = None
            else:
                attn = block.self_attn
                # q_proj 输出 [num_heads, head_dim*2]，HF view(-1, head_dim*2).chunk(2,-1)：
                # 每 head 的 2*head_dim 块 = [query(head_dim) | gate(head_dim)]（交错）。
                # 重排权重行 → [query 全部 | gate 全部]，使 q 段连续（复用 contiguous-seg kernel）。
                # 行重排对 bf16 和 int8 都成立（int8 的 scale 随行一起重排）。
                w_q = self._unpack_linear(attn.q_proj, world_size, rank)  # [2*nh*hd, hidden]
                nh, hd = tc.num_attention_heads, tc.head_dim
                if isinstance(w_q, tuple):
                    w_q = self._reorder_qgate(w_q, nh, hd)
                else:
                    w_q = w_q.view(nh, 2 * hd, -1)
                    w_q = torch.cat([w_q[:, :hd, :].reshape(-1, w_q.shape[-1]),
                                     w_q[:, hd:, :].reshape(-1, w_q.shape[-1])], dim=0).contiguous()
                w_k = self._unpack_linear(attn.k_proj, world_size, rank)
                w_v = self._unpack_linear(attn.v_proj, world_size, rank)
                attn._qkv_w = self._store_w(self._q(self._cat_w([w_q, w_k, w_v])))
                attn._o_w = self._store_w(self._q(self._unpack_linear(attn.o_proj, world_size, rank, chunk_dim=1)))
                attn._q_norm_w = attn.q_norm.weight.data.clone()
                attn._k_norm_w = attn.k_norm.weight.data.clone()
                attn._q_norm_eps = self._ln_eps(attn.q_norm, cfg)
                attn._k_norm_eps = self._ln_eps(attn.k_norm, cfg)
                attn.q_proj = attn.k_proj = attn.v_proj = attn.o_proj = None
                attn.q_norm = attn.k_norm = None

            mlp = block.mlp
            w_up = self._unpack_linear(mlp.up_proj, world_size, rank)
            w_gate = self._unpack_linear(mlp.gate_proj, world_size, rank)
            mlp._gu = self._store_w(self._q(self._cat_w([w_up, w_gate])))
            mlp._d = self._store_w(self._q(self._unpack_linear(mlp.down_proj, world_size, rank, chunk_dim=1)))
            mlp.gate_proj = mlp.up_proj = mlp.down_proj = None

            block._in_ln_w = block.input_layernorm.weight.data.clone()
            block._in_ln_eps = self._ln_eps(block.input_layernorm, cfg)
            block._post_ln_w = block.post_attention_layernorm.weight.data.clone()
            block._post_ln_eps = self._ln_eps(block.post_attention_layernorm, cfg)
            block._is_gdn = is_gdn
            block._prepared = True
        torch.cuda.empty_cache()

        # lm_head int8（Marlin）：bf16 lm_head（2.54GB，quant ignore 存 bf16）→ int8
        # Marlin（1.27GB），forward 走 marlin_forward。每 spec step 被调 2 次（verify
        # M=8 + draft M=7 共享 target lm_head），省 ~3.5ms/step（~6.5%）。全路径对称
        # 走 int8（verify/draft/非spec decode 共享）→ spec==非spec 等价性保持。
        if _LMHEAD_INT8:
            from kernel.marlin import linear_to_marlin
            model.lm_head = linear_to_marlin(model.lm_head)
            torch.cuda.empty_cache()

    # -------------------- GDN 公共 --------------------
    def _gdn_forward(self, la, h2d, graph, bs, is_decode,
                     cu_seqlens=None, seq_idx=None):
        M = h2d.shape[0]
        dev = h2d.device
        H, HK, DK, DV = self._gdn_H, self._gdn_HK, self._gdn_DK, self._gdn_DV
        conv_dim = self._gdn_conv_dim
        scale = DK ** -0.5

        # 输入投影：
        #  - bf16（0.8B）：4 个融合成 1 个 GEMV → [M, 8224]，段布局
        #    qkv[0:6144] | z[6144:8192] | b[8192:8208] | a[8208:8224]（row stride = 8224）。
        #  - W8A16（27B）：qkv/z int8、b/a bf16，dtype 不同不能混融 → 2 个 GEMV：
        #    qkv+z → [M, conv_dim+H*DV]（int8），b+a → [M, 2H]（bf16）。
        # 下游 kernel 用 row-stride 读对应 buffer 的段（qkv/z 视图 / b/a 视图）。
        qz_dim = conv_dim + H * DV
        if getattr(la, "_gdn_w8", False):
            qz = torch.empty(M, qz_dim, dtype=h2d.dtype, device=dev)
            self._lin(h2d, la._in_w_qz, qz, "MICRO_GEMV_GDN")
            ba = torch.empty(M, 2 * H, dtype=h2d.dtype, device=dev)
            if bool(getattr(graph, "_gdn_cp_enabled", False)):
                # verify（M=1+N）：ba 是 bf16 GEMM（b/a 不量化，走 gemv_or_matmul）。
                # decode(M=1) 走 gemv_v2、verify(M=8) 走 torch.matmul——cuBLAS 对 M=8/M=1
                # 选不同 tiling → bf16 归约顺序不同 → 逐行 1-ULP 差（max_abs 0.03125）→
                # 经 g(衰减率)→GDN fp32 递归放大 ~129 步 → argmax 翻转(margin1.75)→ spec
                # target 漂移进退化循环 → mid/long acc 崩塌。逐行 gemv_v2(M=1) 对齐 decode
                # （bitwise 一致）。M 小(≤8)，8 次 launch 在 CUDA graph 内可忽略。
                for _r in range(M):
                    gemv_v2(h2d[_r:_r + 1], la._in_w_ba, ba[_r:_r + 1])
            else:
                self._lin(h2d, la._in_w_ba, ba, "MICRO_GEMV_GDN")
            qkv = qz[:, :conv_dim]
            z = qz[:, conv_dim:]
            b = ba[:, :H]
            a = ba[:, H:]
            in_dim = qz_dim          # qkv/z 视图的 row stride
            ba_stride = 2 * H        # b/a 视图的 row stride
        else:
            in_dim = qz_dim + 2 * H  # 6144 + 2048 + 16 + 16 = 8224
            in_proj = torch.empty(M, in_dim, dtype=h2d.dtype, device=dev)
            self._lin(h2d, la._in_w, in_proj, "MICRO_GEMV_GDN")
            qkv = in_proj[:, :conv_dim]
            z = in_proj[:, conv_dim:conv_dim + H * DV]
            b = in_proj[:, conv_dim + H * DV:conv_dim + H * DV + H]
            a = in_proj[:, conv_dim + H * DV + H:]
            ba_stride = in_dim

        # g（衰减率）HF 全程 fp32（-A_log.float().exp()*softplus），bf16 存会因 exp(g)
        # 在递归里逐 token 复利放大误差 → 改 fp32。beta 是 sigmoid∈[0,1]，bf16 足够。
        g = torch.empty(M, H, dtype=torch.float32, device=dev)
        beta = torch.empty(M, H, dtype=h2d.dtype, device=dev)
        gdn_gbeta(a, b, la._a_log, la._dt_bias, g, beta, stride=ba_stride)

        state = graph._gdn_state_pool
        conv_state = graph._gdn_conv_state_pool

        n_gdn = self._n_gdn
        gdn_l = la._gdn_layer_idx
        if is_decode:
            is_real = graph._gdn_is_real[:bs]
            _gdn_conv_decode_kernel[(bs, triton.cdiv(conv_dim, 512))](
                qkv, la._conv_w, conv_state, graph._gdn_seq_idx[:bs], is_real,
                conv_dim, in_dim, n_gdn, gdn_l, K=self._gdn_K, BLOCK_C=512)
            o = torch.empty(M, H * DV, dtype=h2d.dtype, device=dev)
            _gdn_recurrent_decode_kernel[(bs, H)](
                qkv, g, beta, state, o, graph._gdn_seq_idx[:bs], is_real,
                H=H, HK=HK, DK=DK, DV=DV, N_GDN=n_gdn, GDN_L=gdn_l, SCALE=scale,
                STRIDE=in_dim, BLOCK_D=triton.next_power_of_2(max(DK, DV)))
        else:
            n_seqs = cu_seqlens.shape[0] - 1
            # 投机解码检查点：graph._gdn_cp_enabled 时把每 token 的 conv/recurrent 状态
            # 存进 graph._gdn_cp_conv / _gdn_cp_state（[M, n_gdn, ...]，M=total tokens）。
            # 正常 prefill 不启用（CP_ENABLED=False，kernel 内分支被编译掉，零开销）。
            cp_enabled = bool(getattr(graph, "_gdn_cp_enabled", False))
            cp_state = getattr(graph, "_gdn_cp_state", None)
            cp_conv = getattr(graph, "_gdn_cp_conv", None)
            # spec 去 rollback：非首步 verify 的初始状态直接读 checkpoint[accepted_prev]
            # （graph._gdn_init_state_s/_c 是 [n_gdn, ...] 视图，token 索引已 bake 进 base 指针）。
            # recurrent 是 fp32 [n_gdn,H,DK,DV]、conv 是 bf16 [n_gdn,3,conv_dim]，两个独立视图。
            # 首步 verify / 正常 prefill：INIT_FROM_CP=False，从 pool 读（INIT_STATE 传 pool 占位）。
            init_from_cp = bool(getattr(graph, "_gdn_init_from_cp", False))
            # CUDA graph 安全：INIT_STATE 指向【完整 checkpoint buffer base】
            # [M, n_gdn, ...]，token 索引从 device buffer _gdn_init_idx[0] 读
            # （replay 时重读，非 capture 时 bake 的标量）。旧实现把 token 索引 bake
            # 进切片视图指针（_gdn_cp_state[accepted_prev]），graph replay 会读错行。
            init_idx = graph._gdn_init_idx
            if init_from_cp:
                init_state_s = cp_state                   # [M, n_gdn, H, DK, DV] fp32 base
                init_state_c = cp_conv                     # [M, n_gdn, 3, conv_dim] bf16 base
            else:
                init_state_s = state                       # 占位（kernel 内 INIT_FROM_CP=False 不读）
                init_state_c = conv_state
            if cp_enabled:
                _gdn_conv_prefill_kernel[(triton.cdiv(conv_dim, 512), n_seqs)](
                    qkv, la._conv_w, conv_state, cu_seqlens, seq_idx,
                    conv_dim, in_dim, n_gdn, gdn_l, K=self._gdn_K, BLOCK_C=512,
                    CHECKPOINT=cp_conv, CP_N_GDN=n_gdn, CP_ENABLED=True,
                    INIT_STATE=init_state_c, INIT_IDX=init_idx, INIT_FROM_CP=init_from_cp)
                o = torch.empty(M, H * DV, dtype=h2d.dtype, device=dev)
                _gdn_recurrent_prefill_kernel[(n_seqs, H)](
                    qkv, g, beta, state, o, cu_seqlens, seq_idx,
                    H=H, HK=HK, DK=DK, DV=DV, N_GDN=n_gdn, GDN_L=gdn_l, SCALE=scale,
                    STRIDE=in_dim, BLOCK_D=triton.next_power_of_2(max(DK, DV)),
                    CHECKPOINT=cp_state, CP_N_GDN=n_gdn, CP_ENABLED=True,
                    INIT_STATE=init_state_s, INIT_IDX=init_idx, INIT_FROM_CP=init_from_cp)
            else:
                _gdn_conv_prefill_kernel[(triton.cdiv(conv_dim, 512), n_seqs)](
                    qkv, la._conv_w, conv_state, cu_seqlens, seq_idx,
                    conv_dim, in_dim, n_gdn, gdn_l, K=self._gdn_K, BLOCK_C=512,
                    CHECKPOINT=conv_state, CP_N_GDN=n_gdn, CP_ENABLED=False,
                    INIT_STATE=init_state_c, INIT_IDX=init_idx, INIT_FROM_CP=init_from_cp)
                o = torch.empty(M, H * DV, dtype=h2d.dtype, device=dev)
                _gdn_recurrent_prefill_kernel[(n_seqs, H)](
                    qkv, g, beta, state, o, cu_seqlens, seq_idx,
                    H=H, HK=HK, DK=DK, DV=DV, N_GDN=n_gdn, GDN_L=gdn_l, SCALE=scale,
                    STRIDE=in_dim, BLOCK_D=triton.next_power_of_2(max(DK, DV)),
                    CHECKPOINT=state, CP_N_GDN=n_gdn, CP_ENABLED=False,
                    INIT_STATE=init_state_s, INIT_IDX=init_idx, INIT_FROM_CP=init_from_cp)

        og = torch.empty(M, H * DV, dtype=h2d.dtype, device=dev)
        _gdn_norm_gated_kernel[(M, H)](o, z, la._norm_w, og,
                                       H=H, DV=DV, eps=la._norm_eps,
                                       Z_STRIDE=in_dim,
                                       BLOCK_D=triton.next_power_of_2(DV))
        out = torch.empty(M, h2d.shape[1], dtype=h2d.dtype, device=dev)
        self._lin(og, la._o_w, out, "MICRO_GEMV_GDN")
        return out

    # -------------------- decode 单层钩子 --------------------
    def compute_qkv(self, block, h, graph, bs):
        rmsnorm1_(h, block._in_ln_w, graph._h_buf[:bs], block._in_ln_eps)
        if block._is_gdn:
            return graph._h_buf[:bs]
        attn = block.self_attn
        qkv_buf = graph._qkv[:bs]
        self._lin(graph._h_buf[:bs], attn._qkv_w, qkv_buf, "MICRO_GEMV_QKV")
        return qkv_buf

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        # 返回 (attn_input, residual)：decode 循环 `qkv, h = compute_next_qkv(...)` 解包两值。
        # GDN 层 attn_input = 归一化后的 h（投影延迟到 attention 内做）；full 层 = 投影后 qkv。
        rmsnorm1_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block_next._in_ln_eps
        )
        if block_next._is_gdn:
            return graph._h_buf[:bs], graph._residual[:bs]
        attn = block_next.self_attn
        qkv_buf = graph._qkv[:bs]
        self._lin(graph._h_buf[:bs], attn._qkv_w, qkv_buf, "MICRO_GEMV_QKV")
        return qkv_buf, graph._residual[:bs]

    def attention(self, attn_input, block, layer_idx, bs, graph, cache_manager, block_table):
        if block._is_gdn:
            return self._attention_gdn_decode(attn_input, block, bs, graph)
        return self._attention_full_decode(attn_input, block, layer_idx, bs,
                                           graph, cache_manager, block_table)

    def _attention_full_decode(self, qkv, block, layer_idx, bs, graph,
                               cache_manager, block_table):
        sa = block.self_attn
        nh, kvh, hd = graph.num_heads, graph.kv_num_heads, graph.head_size
        q_dim = nh * hd
        kv_dim = kvh * hd
        k_off = 2 * q_dim  # [q | gate | k | v]
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]

        qk_norm_rope_partial_inplace(
            qkv, bs, 0, nh, hd, sa._q_norm_w,
            graph.attention._cos_pool, graph.attention._sin_pool, cache_lens, sa._q_norm_eps)
        qk_norm_rope_partial_inplace(
            qkv, bs, k_off, kvh, hd, sa._k_norm_w,
            graph.attention._cos_pool, graph.attention._sin_pool, cache_lens, sa._k_norm_eps)

        q = qkv[:, :q_dim].view(bs, nh, hd)
        k = qkv[:, k_off:k_off + kv_dim].view(bs, kvh, hd)
        v = qkv[:, k_off + kv_dim:].view(bs, kvh, hd)

        k_cache, v_cache = cache_manager.get(layer_idx)
        store_kvcache(k, v, k_cache, v_cache, graph._slot_mapping[:bs])
        attn = flash_attn_with_kvcache(
            q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=graph._flash_seqlens[:bs], block_table=block_table,
            causal=True, window_size=(-1, -1), alibi_slopes=None,
            num_splits=0 if bs == 1 else (0 if bs >= 32 else max(1, 32 // max(1, bs * 4)))
        ).squeeze(1)

        # attn_output_gate：gate = q_proj 后半（qkv[:, q_dim:2*q_dim]）
        # Triton in-place（attn *= sigmoid(gate)），替代 sigmoid+cast+mul 三个 elementwise kernel。
        gate = qkv[:, q_dim:2 * q_dim].view(bs, nh, hd)
        attn_gate_inplace(attn, gate)

        out_buf = graph._attn_out[:bs]
        return self._lin(attn.reshape(bs, -1), sa._o_w, out_buf, "MICRO_GEMV_O")

    def _attention_gdn_decode(self, h_normed, block, bs, graph):
        # on_decode_batch 已填 graph._gdn_seq_idx + graph._gdn_is_real（常驻 buffer，
        # graph 安全）。kernel 读 is_real 跳过 pad 行（不重复更新状态）。
        return self._gdn_forward(block.linear_attn, h_normed, graph, bs,
                                 is_decode=True)

    def compute_ffn(self, block, attn_out, residual, graph, bs):
        rmsnorm1_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        mlp_out = dense_swiglu(graph._h_buf[:bs], block.mlp._gu, block.mlp._d, bs, w_is_nk=True)
        return mlp_out, graph._residual[:bs]

    # -------------------- prefill 单层钩子 --------------------
    # 正常 prefill 路径（model_prefill.py / 其他 adapter 共享调用点）：h 是完整残差流，
    # residual=None，返回 h（1 值），行为与 Bug #2 修复前完全一致（pre-attn rmsnorm1(h_bf16)
    # mean_sq bf16 对齐 HF，residual add bf16）。
    def prefill(self, block, h, layer_idx, graph, cache_manager, meta):
        if block._is_gdn:
            return self._prefill_gdn(block, h, None, graph, meta)[2]
        return self._prefill_full(block, h, None, layer_idx, graph, cache_manager, meta)[2]

    # spec verify 路径（spec_decode.py 专用）：Bug #2 修复——层间传 (mlp_out, residual)
    # 分离（不预加 bf16），pre-attention norm 用 fused rmsnorm1_residual（mean_sq 在 fp32
    # mlp_out+residual 上算，对齐 decode compute_next_qkv→rmsnorm1_residual）。原
    # rmsnorm1(h_bf16) 在 bf16 舍入后 residual 上算 mean_sq → 1-ULP 差经 48 GDN 层×129 步
    # 累积 → margin1.75 翻转 → spec target 漂移进循环。返回 (mlp_out, residual, h)：
    # h=(mlp_out+residual) bf16 供 aux 收集 / 逐层 dump / 下一层。
    def prefill_verify(self, block, mlp_out, residual, layer_idx, graph, cache_manager, meta):
        if block._is_gdn:
            return self._prefill_gdn(block, mlp_out, residual, graph, meta)
        return self._prefill_full(block, mlp_out, residual, layer_idx, graph, cache_manager, meta)

    def _prefill_full(self, block, mlp_out, residual, layer_idx, graph, cache_manager, meta):
        sa = block.self_attn
        nh, kvh, hd = graph.num_heads, graph.kv_num_heads, graph.head_size
        q_dim = nh * hd
        kv_dim = kvh * hd
        T = mlp_out.shape[0]

        # pre-attention norm：verify（_gdn_cp_enabled）用 fused（mean_sq 在 fp32
        # mlp_out+residual 上算，对齐 decode compute_next_qkv→rmsnorm1_residual）；正常
        # prefill 用 rmsnorm1(h_bf16)（mean_sq bf16，对齐 HF，行为不变）。layer 0
        # （residual=None）：h=mlp_out（embed），无 residual 可加，用 rmsnorm1。
        if residual is None:
            normed = rmsnorm1(mlp_out, block._in_ln_w, block._in_ln_eps)
            h = mlp_out
        elif bool(getattr(graph, "_gdn_cp_enabled", False)):
            normed, h = rmsnorm1_residual_fused(
                mlp_out, residual, block._in_ln_w, block._in_ln_eps)
        else:
            h = mlp_out + residual
            normed = rmsnorm1(h, block._in_ln_w, block._in_ln_eps)
        qkv = self._lin_prefill(normed, sa._qkv_w)  # [T, 2*q_dim+2*kv_dim]（[q|gate|k|v]）
        k_off = 2 * q_dim
        q = qkv[..., :q_dim].reshape(T, nh, hd).contiguous()
        gate = qkv[..., q_dim:2 * q_dim].reshape(T, nh, hd).contiguous()
        k = qkv[..., k_off:k_off + kv_dim].reshape(T, kvh, hd).contiguous()
        v = qkv[..., k_off + kv_dim:].reshape(T, kvh, hd).contiguous()

        q = rmsnorm1(q, sa._q_norm_w, sa._q_norm_eps)
        k = rmsnorm1(k, sa._k_norm_w, sa._k_norm_eps)

        # partial RoPE：Triton in-place（前 rot 维 half-split，rot 后不动），替代 PyTorch
        # 的 cos/sin gather + 4 slice + 4 mul + 2 add + 2 cat（每张量 ~12 小 kernel）。
        cos_pool = graph.attention._cos_pool
        sin_pool = graph.attention._sin_pool
        pos = meta.position_ids.long()
        rope_partial_inplace(q, cos_pool, sin_pool, pos)
        rope_partial_inplace(k, cos_pool, sin_pool, pos)

        k_cache, v_cache = cache_manager.get(layer_idx)
        store_kvcache(k, v, k_cache, v_cache, meta.slot_mapping)
        attn = flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=meta.cu_seqlens_q, cu_seqlens_k=meta.cu_seqlens_k,
            max_seqlen_q=meta.max_seqlen_q, max_seqlen_k=meta.max_seqlen_k,
            softmax_scale=hd ** -0.5, causal=True,
            block_table=meta.block_table,
        )
        # attn_output_gate：Triton in-place（attn *= sigmoid(gate)），替代 sigmoid+cast+mul。
        attn_gate_inplace(attn, gate)
        out = self._lin_prefill(attn.reshape(T, -1), sa._o_w)

        normed, residual = rmsnorm1_residual_fused(out, h, block._post_ln_w, block._post_ln_eps)
        mlp_out = dense_swiglu(normed, block.mlp._gu, block.mlp._d, T, w_is_nk=True)
        # 返回 (mlp_out, residual, R_L)：R_L 是下一层残差流。verify 用 fp32 add（对齐
        # decode 非末层 rmsnorm1_residual 的 fp32 add→round bf16）；正常 prefill 用 bf16
        # PyTorch add（对齐 HF，64-token HF 对齐不受影响）。
        if bool(getattr(graph, "_gdn_cp_enabled", False)):
            R_L = (mlp_out.float() + residual.float()).to(mlp_out.dtype)
        else:
            R_L = mlp_out + residual
        return mlp_out, residual, R_L

    def _prefill_gdn(self, block, mlp_out, residual, graph, meta):
        # 完整一层（对齐 _prefill_full / HF Qwen3_5DecoderLayer）。Bug #2 修复：层间传
        # (mlp_out, residual) 分离，pre-attention norm 用 fused（mean_sq fp32，对齐 decode）。
        #   h = mlp_out + residual          # 残差流（layer 0: h=mlp_out=embed）
        #   normed = rmsnorm1_residual_fused(mlp_out, residual)  # pre-attn norm（fp32 mean_sq）
        #   gdn_out = GDN(normed)
        #   res_after_attn = h + gdn_out    # post-attn 残差
        #   mlp_out_L = mlp(rmsnorm1_residual_fused(gdn_out, h))
        #   返回 (mlp_out_L, res_after_attn, R_L)：R_L = mlp_out_L + res_after_attn（bf16）
        n_seqs = meta.n_seqs
        seq_idx = graph._gdn_prefill_seq_idx[:n_seqs]
        cu = meta.cu_seqlens_q
        T = mlp_out.shape[0]
        if residual is None:
            normed = rmsnorm1(mlp_out, block._in_ln_w, block._in_ln_eps)
            h = mlp_out
        else:
            normed, h = rmsnorm1_residual_fused(
                mlp_out, residual, block._in_ln_w, block._in_ln_eps)
        gdn_out = self._gdn_forward(block.linear_attn, normed, graph, T,
                                    is_decode=False, cu_seqlens=cu, seq_idx=seq_idx)
        normed2, res_after_attn = rmsnorm1_residual_fused(
            gdn_out, h, block._post_ln_w, block._post_ln_eps)
        mlp_out_L = dense_swiglu(normed2, block.mlp._gu, block.mlp._d, T, w_is_nk=True)
        # R_L 下一层残差流：verify 用 fp32 add（对齐 decode）；正常 prefill 用 bf16 PyTorch
        # add（对齐 HF，64-token HF 对齐不受影响）。
        if bool(getattr(graph, "_gdn_cp_enabled", False)):
            R_L = (mlp_out_L.float() + res_after_attn.float()).to(mlp_out_L.dtype)
        else:
            R_L = mlp_out_L + res_after_attn
        return mlp_out_L, res_after_attn, R_L

    # -------------------- buffer 分配 --------------------
    # GDN 状态池是【类级单例】：prefill runner 与 decode runner 是独立实例（各自
    # build_adapter + alloc_bufs），但 GDN 递归/conv 状态必须跨两者共享（prefill 建
    # 状态、decode 续写）。用类属性按 device 缓存。
    #
    # 池大小 = max_bs（并发序列上限）。每 seq 占 n_gdn 层 × H×DK×DV fp32（0.8B:
    # 18×16×128×128×4B ≈ 18MB/seq）。max_bs=64 → ~1.2GB，max_bs=512 → ~9.5GB。
    # slot 分配器（free list + in_use）也是类级：prefill 首 chunk 分配 slot、
    # seq 完成时释放，避免 seq_id % POOL 的碰撞（多 seq 并发时 mod 会撞）。
    _shared = {}

    @staticmethod
    def _dev_key(device):
        """device 规范化为 'cuda:N'（alloc_bufs 传 'cuda' 无 index，hook 里 tensor.device
        是 'cuda:0'，两者须归一到同一 key 才能命中同一状态池）。"""
        d = torch.device(device)
        if d.type == "cuda" and d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        return str(d)

    @classmethod
    def _get_shared(cls, n_gdn, H, DK, DV, conv_dim, K, pool_size, dtype, device):
        key = cls._dev_key(device)
        if key not in cls._shared:
            cls._shared[key] = {
                "state": torch.zeros(pool_size, n_gdn, H, DK, DV,
                                     dtype=torch.float32, device=device),
                "conv": torch.zeros(pool_size, n_gdn, K - 1, conv_dim,
                                    dtype=dtype, device=device),
                "free": list(range(pool_size - 1, -1, -1)),  # slot 栈
                "in_use": {},  # seq_id -> slot
            }
        return cls._shared[key]

    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        tc = self._tcfg
        qkv_dim = tc.num_attention_heads * tc.head_dim * 2 + 2 * tc.num_key_value_heads * tc.head_dim
        H, DK, DV = self._gdn_H, self._gdn_DK, self._gdn_DV
        shared = self._get_shared(self._n_gdn, H, DK, DV, self._gdn_conv_dim,
                                  self._gdn_K, max_bs, dtype, device)
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, qkv_dim, dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, hidden_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_gdn_state_pool": shared["state"],
            "_gdn_conv_state_pool": shared["conv"],
            "_gdn_seq_idx": torch.zeros(max_bs, dtype=torch.int32, device=device),
            "_gdn_is_real": torch.zeros(max_bs, dtype=torch.int32, device=device),
            "_gdn_prefill_seq_idx": torch.zeros(max_bs, dtype=torch.int32, device=device),
            # CUDA graph 安全：GDN 初始状态 token 索引（device buffer，replay 时重读）。
            # spec verify 非首步从 checkpoint[accepted_prev] 读初始状态，token 索引
            # 每步变，不能 bake 进指针 → 存 buffer[0]，kernel 内 tl.load 读。
            "_gdn_init_idx": torch.zeros(1, dtype=torch.int32, device=device),
        }

    # -------------------- 有状态层 batch 元信息 --------------------
    def gdn_stateful(self) -> bool:
        return self._n_gdn > 0

    def _alloc_slot(self, seq, shared):
        """给 seq 分配一个 GDN 状态池 slot（存到 seq._gdn_slot）。幂等。"""
        if getattr(seq, "_gdn_slot", None) is not None:
            return seq._gdn_slot
        if not shared["free"]:
            raise RuntimeError(
                "GDN 状态池耗尽（并发序列 > 池大小）。max_batch_size 需 ≥ 并发数。")
        slot = shared["free"].pop()
        shared["in_use"][seq.seq_id] = slot
        seq._gdn_slot = slot
        return slot

    def on_decode_batch(self, batch, graph):
        """decode：batch 是 pad 后列表（循环复制）。真实 seq = 首次出现的行。
        填常驻 buffer（graph 安全：kernel 读 buffer，replay 时重读）：
        - _gdn_seq_idx[row] = 该行 seq 的 state 池 slot（pad 行填 0）
        - _gdn_is_real[row] = 1 真实行 / 0 pad 行（kernel 据此跳过 pad 行，
          避免同一 seq 状态被 pad 副本重复更新）"""
        shared = self._shared[self._dev_key(graph._gdn_seq_idx.device)]
        dev = graph._gdn_seq_idx.device
        seen = set()
        seq_idx = torch.zeros(len(batch), dtype=torch.int32)
        is_real = torch.zeros(len(batch), dtype=torch.int32)
        for i, seq in enumerate(batch):
            if seq.seq_id in seen:
                continue
            seen.add(seq.seq_id)
            slot = self._alloc_slot(seq, shared)
            seq_idx[i] = slot
            is_real[i] = 1
        graph._gdn_seq_idx[:len(batch)] = seq_idx.to(dev)
        graph._gdn_is_real[:len(batch)] = is_real.to(dev)

    def on_prefill_batch(self, batch, graph):
        """prefill：分配 slot + 填状态池索引；首 chunk（prefill_done==0）清零该 seq
        的 GDN 状态（新序列从空状态开始；chunked 续写复用已有状态）。"""
        shared = self._shared[self._dev_key(graph._gdn_prefill_seq_idx.device)]
        n = len(batch)
        idx = torch.zeros(n, dtype=torch.int32)
        for i, s in enumerate(batch):
            idx[i] = self._alloc_slot(s, shared)
        graph._gdn_prefill_seq_idx[:n] = idx.to(graph._gdn_prefill_seq_idx.device)
        # 首 chunk 清零 recurrent + conv 状态
        fresh = [i for i, s in enumerate(batch) if s.prefill_done == 0]
        if fresh:
            fidx = idx[fresh]
            graph._gdn_state_pool[fidx] = 0
            graph._gdn_conv_state_pool[fidx] = 0

    def on_seq_finished(self, seq):
        """seq 完成：释放其 GDN 状态池 slot。"""
        slot = getattr(seq, "_gdn_slot", None)
        if slot is None:
            return
        for shared in self._shared.values():
            if seq.seq_id in shared["in_use"]:
                shared["in_use"].pop(seq.seq_id)
                shared["free"].append(slot)
                break
        seq._gdn_slot = None
