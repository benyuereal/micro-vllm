"""Marlin W8A16 int8 GEMM（CUTLASS C++，int8 权重直接进 int8 tensor-core mma）。

从 vLLM csrc/libtorch_stable/quantization/marlin 移植（stable-ABI → 标准
torch/extension.h），用 micro-vllm 的 load_inline 编译体系（同 gemv_int8.py）。

权重格式（checkpoint 是 Marlin 格式）：
  - weight_packed int32 [N, K/4]（每 int32 打包 4 个 int8，byte i = (int8+128)&0xFF）
  - weight_scale bf16 [N, K/128]（group-128）

加载时（build_marlin）：
  1. packed [N,K/4] → .t().contiguous() 转 [K/4, N]（Marlin packed_dim=0）
  2. marlin_pad_qweight（tile 对齐 padding）
  3. gptq_marlin_repack（CUDA kernel，重排成 Marlin 布局）
  4. scale → marlin_pad_scales + marlin_permute_scales
  5. 存 (wq, ws, g_idx, zp, workspace, padded_n, padded_k, N, K)

forward（marlin_forward）：
  x [M,K] bf16 → marlin_pad_dim → ops.marlin_gemm(uint8b128, num_bits=8, is_k_full)
  → marlin_unpad_output → [M,N] bf16

数值：int8 权重 int32 accumulate（精确），bf16 激活，与反量化 matmul 有 bf16 级
误差（<0.1 maxdiff），与 TileLang 版本同量级。
"""
import os
import logging

import torch

logger = logging.getLogger(__name__)

_mod = None
_init_err = None

# Marlin 常量（同 vLLM marlin_utils）
MARLIN_TILE = 16
MARLIN_MIN_THREAD_N = 64
MARLIN_MIN_THREAD_K = 128
GROUP = 128


def _load():
    global _mod, _init_err
    if _mod is not None or _init_err is not None:
        return
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _mod = load(
            name="micro_marlin",
            sources=[
                os.path.join(here, "marlin.cu"),
                os.path.join(here, "gptq_marlin_repack.cu"),
                os.path.join(here, "sm80_kernel_bfloat16_u8b128_bfloat16.cu"),
                os.path.join(here, "sm80_kernel_float16_u8b128_float16.cu"),
            ],
            extra_cuda_cflags=[
                "-O3", "-arch=sm_89", "--use_fast_math",
                "-gencode=arch=compute_89,code=sm_89",
                # Marlin 模板实例化在 sm80_*.cu（显式实例化），marlin.cu 的
                # get_marlin_kernel 跨 TU 引用。默认 -static-global-template-stub=true
                # 会把跨 TU 模板引用标成 hidden → 链接 undefined reference。关掉后
                # 保留外部引用，链接器解析到 sm80 文件的显式实例化符号。
                "-static-global-template-stub=false",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Marlin int8 GEMM kernel 编译成功（W8A16 CUTLASS C++）")
    except Exception as e:
        _init_err = e
        logger.warning(f"Marlin int8 GEMM 编译失败: {e}")


def marlin_available() -> bool:
    if _mod is None and _init_err is None:
        _load()
    return _mod is not None


# ---------------------------------------------------------------------------
# 权重预处理（加载时一次）
# ---------------------------------------------------------------------------
def _round_up(x, m):
    return (x + m - 1) // m * m


def marlin_padded_nk(size_n, size_k, group_size=GROUP):
    """最小 (padded_n, padded_k) 满足 Marlin thread-tile 族。
    Marlin 要求 (n%64, k%128) 或 (n%128, k%64)；都不满足时零 padding 到更便宜的族。"""
    import math
    group = group_size if group_size > 0 else 1
    candidates = (
        (_round_up(size_n, 64), _round_up(size_k, math.lcm(128, group))),
        (_round_up(size_n, 128), _round_up(size_k, math.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k):
    """零 padding GPTQ 布局 packed 权重 (size_k/pack, size_n) 供 gptq_marlin_repack。"""
    if (padded_n, padded_k) == (size_n, size_k):
        return qweight
    pack_factor = size_k // qweight.size(0)
    return torch.nn.functional.pad(
        qweight, (0, padded_n - size_n, 0, (padded_k - size_k) // pack_factor))


def marlin_pad_scales(scales, size_n, size_k, padded_n, padded_k, group_size):
    """零 padding 权重 scale (num_groups, size_n)；在 marlin_permute_scales 前调用。"""
    if (padded_n, padded_k) == (size_n, size_k):
        return scales
    pad_rows = padded_k // group_size - scales.size(0) if group_size > 0 else 0
    assert pad_rows >= 0
    return torch.nn.functional.pad(
        scales, (0, padded_n - size_n, 0, pad_rows))


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


def marlin_pad_dim(x, size, padded):
    """零 padding 最后一维（激活 K / bias N）。"""
    if padded == size:
        return x
    return torch.nn.functional.pad(x, (0, padded - size))


def marlin_unpad_output(output, size_n, padded_n):
    """去掉 padding 输出列，回到逻辑 N。"""
    if padded_n == size_n:
        return output
    return output[..., :size_n].contiguous()


def marlin_make_workspace(device):
    """Marlin workspace（num threadblocks = sms * max_blocks_per_sm）。"""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int, device=device, requires_grad=False)


def marlin_make_empty_g_idx(device):
    return torch.empty(0, dtype=torch.int, device=device)


def build_marlin(packed, scale_bf16, N, K, device):
    """checkpoint weight_packed int32 [N,K/4] + weight_scale bf16 [N,K/128] → Marlin 格式。

    复刻 vLLM MarlinLinearKernel.process_weights_after_loading（无 act-order、无 zp）。
    返回 dict(wq, ws, g_idx, zp, workspace, padded_n, padded_k, N, K)。
    """
    assert marlin_available(), "Marlin kernel 不可用"
    # 我们 checkpoint 布局 [N, K/4]/[N, K/128]；vLLM Marlin 布局 packed_dim=0 即 [K/4, N]
    wq = packed.t().contiguous().to(device)
    ws = scale_bf16.t().contiguous().to(device)
    padded_n, padded_k = marlin_padded_nk(N, K, GROUP)
    wq = marlin_pad_qweight(wq, N, K, padded_n, padded_k)
    wq = _mod.gptq_marlin_repack(
        wq, marlin_make_empty_g_idx(device),
        size_k=padded_k, size_n=padded_n, num_bits=8, is_a_8bit=False)
    ws = marlin_permute_scales(
        marlin_pad_scales(ws, N, K, padded_n, padded_k, GROUP),
        size_k=padded_k, size_n=padded_n, group_size=GROUP, is_a_8bit=False)
    g_idx = marlin_make_empty_g_idx(device)
    zp = marlin_make_empty_g_idx(device)
    workspace = marlin_make_workspace(device)
    return dict(wq=wq, ws=ws, g_idx=g_idx, zp=zp, workspace=workspace,
                padded_n=padded_n, padded_k=padded_k, N=N, K=K)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def marlin_forward(m, x):
    """x bf16 [M,K] → bf16 [M,N]（复刻 apply_gptq_marlin_linear 调用）。

    m 是 build_marlin 返回的 dict。"""
    xp = marlin_pad_dim(x, m["K"], m["padded_k"])
    out = _mod.marlin_gemm(
        xp, None, m["wq"], None, m["ws"], None, None, m["zp"], m["g_idx"],
        m["g_idx"], m["workspace"], _mod.marlin_u8b128_id(),
        size_m=xp.shape[0], size_n=m["padded_n"], size_k=m["padded_k"],
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=False,
        is_zp_float=False)
    return marlin_unpad_output(out, m["N"], m["padded_n"])
