"""Marlin W4A16 GPTQ MoE 算子（CUTLASS C++，int4 权重寄存器反量化 + fp16 mma）。

从 vLLM csrc/libtorch_stable/moe/marlin_moe_wna16 移植（stable-ABI → 标准
torch/extension.h），用 micro-vllm 的 load 编译体系（同 kernel/marlin）。

与 kernel/marlin（W8A16 单 GEMM）的区别：这是 MoE 版，一次 GEMM 处理多个 expert
（按 sorted_token_ids 分块，每块一个 expert），支持 top-k 路由 + router 权重。

权重格式（GPTQ W4，group-128，有 zero-point）：
  gate_up: qweight [E, H//8, 2*INTER] int32（8 int4/int32，低 nibble=偶 k），
           scales [E, H//128, 2*INTER] bf16，qzeros [E, H//128, 2*INTER//8] int32
  down:    qweight [E, INTER//8, H] int32，scales [E, INTER//128, H] bf16，
           qzeros [E, INTER//128, H//8] int32

build_marlin_moe（加载时一次）：
  1. 每 expert gptq_marlin_repack（GPTQ packed → Marlin 布局 [E, K//16, N*2]）
  2. scales → marlin_permute_scales（64 元素 perm）
  3. qzeros → unpack + marlin_zero_points（perm + interleave + repack）

moe_marlin_forward（decode 热路径）：
  gate+softmax+topk → moe_align_block_size（纯 torch 路由对齐）
  → gate_up GEMM（size_m=M, top_k=K, mul_topk_weights=False）
  → silu_and_mul → down GEMM（size_m=M*K, top_k=1, mul_topk_weights=True）
  → topk reduce（sum over k）→ [M, H]

数值：int4 权重 int32 accumulate（精确），bf16 激活，与 fp32 反量化 matmul 有
bf16 级误差。W4 权重读是 bf16 的 1/4，memory-bound decode 下理论 ~2x。
"""
import os
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_mod = None
_init_err = None

# Marlin 常量（同 vLLM marlin_utils）
MARLIN_TILE = 16
MARLIN_MIN_THREAD_N = 64
GROUP = 128


def _load():
    global _mod, _init_err
    if _mod is not None or _init_err is not None:
        return
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _mod = load(
            name="micro_marlin_moe",
            sources=[
                os.path.join(here, "moe_ops.cu"),
                os.path.join(here, "gptq_marlin_repack.cu"),
                os.path.join(here, "sm80_moe_kernel_bfloat16_u4_bfloat16.cu"),
                os.path.join(here, "sm80_moe_kernel_bfloat16_u8b128_bfloat16.cu"),
            ],
            extra_cuda_cflags=[
                "-O3", "-arch=sm_89", "--use_fast_math",
                "-gencode=arch=compute_89,code=sm_89",
                # Marlin 模板显式实例化在 sm80_*.cu，moe_ops.cu 的 marlin_mm 跨 TU
                # 引用。默认 -static-global-template-stub=true 会把跨 TU 模板引用标成
                # hidden → 链接 undefined reference。关掉后保留外部引用。
                "-static-global-template-stub=false",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Marlin MoE W4A16 kernel 编译成功（CUTLASS C++）")
    except Exception as e:
        _init_err = e
        logger.warning(f"Marlin MoE W4A16 kernel 编译失败: {e}")


def marlin_moe_available() -> bool:
    if _mod is None and _init_err is None:
        _load()
    return _mod is not None


# ---------------------------------------------------------------------------
# GPTQ W4 量化（合成权重，供测试；真实 checkpoint 直接读 GPTQ 张量）
# ---------------------------------------------------------------------------
def gptq_quantize(W, group=GROUP):
    """bf16/fp32 [N, K] → GPTQ W4 标准布局 + fp32 反量化参考。

    返回 (qweight, scales, qzeros, Wdq)：
      qweight [K//8, N] int32（8 int4/int32，低 nibble=偶 k）
      scales  [K//group, N] bf16
      qzeros  [K//group, N//8] int32（沿 N pack，8 int4/int32）
      Wdq     [N, K] fp32（dequant reference，w=(q-z)*s）
    """
    N, K = W.shape
    assert K % group == 0, f"K={K} 非 group={group} 倍数"
    Wf = W.float()
    Wg = Wf.view(N, K // group, group)
    smin = Wg.amin(dim=2, keepdim=True)
    smax = Wg.amax(dim=2, keepdim=True)
    s = ((smax - smin) / 15.0).clamp_min(1e-8)          # [N, Ng, 1]
    z = ((smin / s).round()).clamp(0, 15)               # [N, Ng, 1]
    q = (((Wg - smin) / s).round()).clamp(0, 15)        # [N, Ng, group]
    q = q.view(N, K).to(torch.int32)                    # [N, K]
    z = z.view(N, K // group).to(torch.int32)           # [N, Ng]
    s = s.view(N, K // group)                           # [N, Ng]
    # qweight [K//8, N]：q[n,k] → qweight[k//8, n] bit (k%8)*4
    q8 = q.view(N, K // 8, 8)
    qweight = torch.zeros(N, K // 8, dtype=torch.int32, device=W.device)
    for j in range(8):
        qweight |= (q8[:, :, j] << (4 * j))
    qweight = qweight.t().contiguous()
    # qzeros [Ng, N//8]：z[n,g] → qzeros[g, n//8] bit (n%8)*4
    z8 = z.view(N // 8, 8, K // group)
    qzeros = torch.zeros(N // 8, K // group, dtype=torch.int32, device=W.device)
    for j in range(8):
        qzeros |= (z8[:, j, :] << (4 * j))
    qzeros = qzeros.t().contiguous()
    # scales [Ng, N]
    scales = s.t().contiguous().to(torch.bfloat16)
    # dequant reference
    Wdq = (q.float() - z.repeat_interleave(group, dim=1).float()) \
        * s.repeat_interleave(group, dim=1)
    return qweight, scales, qzeros, Wdq.view(N, K)


# ---------------------------------------------------------------------------
# Marlin 权重预处理（加载时一次）
# ---------------------------------------------------------------------------
def _round_up(x, m):
    return (x + m - 1) // m * m


def _get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s, size_k, size_n, group_size, is_a_8bit=False):
    """Marlin scale 重排（group-128 用 64 元素 perm，否则 32 元素 single perm）。"""
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def _pack_cols(q_w, num_bits, size_k, size_n):
    """[size_k, size_n] int32（值 0-15）→ [size_k, size_n//pack] int32（沿 N pack）。"""
    pack_factor = 32 // num_bits
    q_res = torch.zeros(size_k, size_n // pack_factor, dtype=torch.int32, device=q_w.device)
    for i in range(pack_factor):
        q_res |= (q_w[:, i::pack_factor] << (num_bits * i))
    return q_res


def _unpack_cols(packed, num_bits, size_k, size_n):
    """[size_k, size_n//pack] int32 → [size_k, size_n] int32（沿 N unpack）。"""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    q_res = torch.zeros(size_k, size_n, dtype=torch.int32, device=packed.device)
    p = packed.clone()
    for i in range(pack_factor):
        q_res[:, i::pack_factor] = p & mask
        p = p >> num_bits
    return q_res


def marlin_zero_points(zp, size_k, size_n, num_bits):
    """[size_k, size_n] int32（unpacked 0-15）→ [size_k, size_n//pack] int32（Marlin zp）。

    scale_perm（64 元素）+ interleave（W4: [0,2,4,6,1,3,5,7]）+ 沿 N repack。"""
    scale_perm, _ = _get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if num_bits == 4 else [0, 2, 1, 3]
    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    return _pack_cols(zp, num_bits, size_k, size_n)


def _repack_moe(qweight, size_k, size_n, num_bits):
    """每 expert gptq_marlin_repack：[E, size_k//pack, size_n] → [E, size_k//16, size_n*2]。"""
    E = qweight.shape[0]
    perm = torch.empty(0, dtype=torch.int32, device=qweight.device)
    out = torch.empty(E, size_k // 16, size_n * (num_bits // 2),
                      dtype=torch.int32, device=qweight.device)
    for e in range(E):
        out[e] = _mod.gptq_marlin_repack(qweight[e], perm, size_k, size_n, num_bits, False)
    return out


def _permute_scales_moe(scales, size_k, size_n, group_size):
    """每 expert marlin_permute_scales：[E, size_k//group, size_n] → 同形状（permuted）。"""
    E = scales.shape[0]
    out = torch.empty_like(scales)
    for e in range(E):
        out[e] = marlin_permute_scales(scales[e], size_k, size_n, group_size)
    return out


def _zero_points_moe(qzeros, num_groups, size_n, num_bits):
    """每 expert：GPTQ packed qzeros [E, num_groups, size_n//pack] → Marlin zp 同形状。"""
    E = qzeros.shape[0]
    out = torch.empty_like(qzeros)
    for e in range(E):
        q_zp = _unpack_cols(qzeros[e], num_bits, num_groups, size_n)
        out[e] = marlin_zero_points(q_zp, num_groups, size_n, num_bits)
    return out


def build_marlin_moe(gu_qweight, gu_scales, gu_qzeros,
                     d_qweight, d_scales, d_qzeros,
                     H, INTER, E, group=GROUP, device="cuda"):
    """GPTQ W4 MoE 权重 → Marlin 格式（加载时一次）。

    gate_up: qweight [E, H//8, 2*INTER], scales [E, H//128, 2*INTER], qzeros [E, H//128, 2*INTER//8]
    down:    qweight [E, INTER//8, H], scales [E, INTER//128, H], qzeros [E, INTER//128, H//8]
    返回 dict(gu={wq,scales,zeros,workspace,b_type_id,N,K}, d={...}, H, INTER, E)。
    """
    assert marlin_moe_available(), "Marlin MoE kernel 不可用"
    assert INTER % 64 == 0, f"INTER={INTER} 非 64 倍数（Marlin thread-tile 要求）"
    assert INTER % group == 0, f"INTER={INTER} 非 group={group} 倍数"
    assert H % 16 == 0, f"H={H} 非 16 倍数"
    sms = torch.cuda.get_device_properties(torch.device(device)).multi_processor_count
    workspace = torch.zeros(sms * 4, dtype=torch.int32, device=device)
    b_type_id = _mod.moe_marlin_u4_id()
    # gate_up: N=2*INTER, K=H
    gu = dict(
        wq=_repack_moe(gu_qweight, H, 2 * INTER, 4),
        scales=_permute_scales_moe(gu_scales, H, 2 * INTER, group),
        zeros=_zero_points_moe(gu_qzeros, H // group, 2 * INTER, 4),
        workspace=workspace, b_type_id=b_type_id, N=2 * INTER, K=H,
    )
    # down: N=H, K=INTER
    d = dict(
        wq=_repack_moe(d_qweight, INTER, H, 4),
        scales=_permute_scales_moe(d_scales, INTER, H, group),
        zeros=_zero_points_moe(d_qzeros, INTER // group, H, 4),
        workspace=workspace, b_type_id=b_type_id, N=H, K=INTER,
    )
    return dict(gu=gu, d=d, H=H, INTER=INTER, E=E)


# ---------------------------------------------------------------------------
# 路由对齐（纯 torch，替代 vLLM C++ moe_align_block_size kernel）
# ---------------------------------------------------------------------------
def moe_align_block_size(topk_ids, block_size, num_experts):
    """topk_ids [M, top_k] → (sorted_token_ids, expert_ids, num_tokens_post_padded)。

    单 block CUDA kernel（moe_ops.cu）。sorted_token_ids 存展平 topk 位置
    [0, M*top_k)（值 = t*top_k+k），padding 填 M*top_k（invalid）。expert_ids 每
    block 一个 expert id。expert 内顺序不影响正确性（每 token 独立）。
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device
    total = M * top_k
    max_padded = total + num_experts * (block_size - 1)
    max_blocks = (max_padded + block_size - 1) // block_size
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    ntp = torch.empty(1, dtype=torch.int32, device=device)
    _mod.moe_align_block_size(topk_ids, block_size, num_experts,
                              sorted_ids, expert_ids, ntp)
    return sorted_ids, expert_ids, ntp


def _pick_block_size(M, top_k, E):
    """vLLM 启发式：M*top_k/E/block_size < 0.9 时取该 block_size（decode 小 M → 16）。"""
    bs = 64
    for b in [8, 16, 32, 48, 64]:
        if M * top_k / E / b < 0.9:
            bs = b
            break
    return max(bs, 16)


# ---------------------------------------------------------------------------
# forward（decode 热路径）
# ---------------------------------------------------------------------------
_routing_cache = {}


def _fused_routing(x, gate_w, n_experts, top_k):
    """gate GEMV + softmax + topk（复用 bf16 moe.py 的融合 TileLang kernel，37us）。

    返回 (topk_idx [M,K] int32, topk_w [M*K] float32)。权重格式无关（只吃 gate_w）。
    """
    M, H = x.shape
    key = (M, H, n_experts, top_k, x.dtype)
    if key not in _routing_cache:
        from kernel.moe import gate_gemv_kernel, softmax_topk_kernel
        import tilelang.language as T
        _routing_cache[key] = (
            gate_gemv_kernel(M, H, n_experts, T.bfloat16),
            softmax_topk_kernel(M, n_experts, top_k, T.bfloat16),
        )
    k_gv, k_st = _routing_cache[key]
    logits = k_gv(x, gate_w)                              # [M, E] bf16
    topk_idx, topk_w = k_st(logits)                       # [M,K] int32, [M,K] bf16
    return topk_idx, topk_w.reshape(-1).float().contiguous()


def moe_marlin_forward(x, gate_w, m, top_k, n_experts):
    """完整 MoE decode（Marlin W4A16）。

    x: [M, H] bf16, gate_w: [E, H] bf16, m: build_marlin_moe 返回的 dict。
    返回 [M, H] bf16。

    流程：
      1. 融合路由（gate GEMV + softmax + topk，复用 bf16 TileLang kernel）
      2. moe_align_block_size（单 block CUDA kernel）
      3. gate_up GEMM（size_m=M, top_k=K, mul_topk_weights=False）→ [M*K, 2*INTER]
      4. silu_and_mul → [M*K, INTER]
      5. down GEMM（size_m=M*K, top_k=1, mul_topk_weights=True）→ [M*K, H]
      6. topk reduce（sum over k）→ [M, H]
    """
    M, H = x.shape
    INTER = m["INTER"]
    # 1. 融合路由
    topk_idx, topk_w = _fused_routing(x, gate_w, n_experts, top_k)
    # 2. 路由对齐
    block_size = _pick_block_size(M, top_k, n_experts)
    sorted_ids, expert_ids, ntp = moe_align_block_size(topk_idx, block_size, n_experts)
    gu, d = m["gu"], m["d"]
    # 3. gate_up GEMM（size_m=M, top_k=K）
    gu_out = _mod.moe_wna16_marlin_gemm(
        x, None, gu["wq"], None, gu["scales"], None, None, gu["zeros"],
        None, None, gu["workspace"], sorted_ids, expert_ids, ntp, topk_w,
        block_size, top_k, False, gu["b_type_id"],
        M, 2 * INTER, H, True, False, True, False, -1, -1, -1)
    # gu_out: [M*K, 2*INTER] token-major（c[t*K+k] = token t 的 k-th expert gate_up）
    # 4. silu_and_mul
    guv = gu_out.view(M * top_k, 2, INTER)
    act = F.silu(guv[:, 0, :]) * guv[:, 1, :]             # [M*K, INTER]
    # 5. down GEMM（size_m=M*K, top_k=1, mul_topk_weights=True）
    d_out = _mod.moe_wna16_marlin_gemm(
        act, None, d["wq"], None, d["scales"], None, None, d["zeros"],
        None, None, d["workspace"], sorted_ids, expert_ids, ntp, topk_w,
        block_size, 1, True, d["b_type_id"],
        M * top_k, H, INTER, True, False, True, False, -1, -1, -1)
    # d_out: [M*K, H] token-major（c[t*K+k] = token t 的 k-th expert down，已乘 router 权重）
    # 6. topk reduce
    out = d_out.view(M, top_k, H).sum(1)                  # [M, H]
    return out
