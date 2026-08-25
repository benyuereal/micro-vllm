"""W8A16 int8 GEMV 加载 + 统一分派。

gemv_int8(x, w_int8, scale, out)：x [M,K] bf16, w_int8 [N,K] int8, scale [N] fp32,
out [M,N] bf16。M=1 走手写 int8 GEMV（权重带宽减半）；M>1 由 w8_linear 反量化后 matmul。
"""
import os
import logging
import torch

logger = logging.getLogger(__name__)

_mod = None
_init_err = None


def _load():
    global _mod, _init_err
    if _mod is not None or _init_err is not None:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        cu_path = os.path.join(os.path.dirname(__file__), "gemv_int8.cu")
        with open(cu_path) as f:
            cuda_src = f.read()
        cpp_src = ("torch::Tensor gemv_int8(torch::Tensor x, torch::Tensor w_int8, "
                   "torch::Tensor scale, torch::Tensor out);\n"
                   "torch::Tensor gemv_int8_group(torch::Tensor x, torch::Tensor w_int8, "
                   "torch::Tensor scale, torch::Tensor out);")
        _mod = load_inline(
            name="micro_gemv_int8",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gemv_int8", "gemv_int8_group"],
            extra_cuda_cflags=["-O3", "-arch=sm_89"],
            verbose=False,
        )
        logger.info("gemv_int8 kernel 编译成功（W8A16 int8 GEMV）")
    except Exception as e:
        _init_err = e
        logger.warning(f"gemv_int8 编译失败，fallback 反量化 matmul: {e}")


def gemv_int8_available() -> bool:
    if _mod is None and _init_err is None:
        _load()
    return _mod is not None


def w8_linear(x, w_int8, scale, out=None, env="MICRO_GEMV"):
    """W8A16 线性层：x [M,K] bf16, w_int8 [N,K] int8, scale fp32 → out [M,N] bf16。

    scale 两种：
      - [N]（per-channel，0.8B 自量化）→ gemv_int8
      - [N, K/128]（group-128，Qwen3.8 预量化）→ gemv_int8_group
    M=1 且 int8 kernel 可用且 env 开启 → 手写 int8 GEMV（权重带宽减半）；
    否则反量化 int8→bf16 后 x @ w.t()（prefill M>1 走此路，compute-bound 反量化开销可忽略）。
    投机解码 verify（M≤8，verify_gemm 开，group-128）→ 双后端 int8 GEMM
    （TileLang 默认 / Triton 备选，MICRO_VERIFY_GEMM 切换），权重 HBM 只读一次。
    """
    M = x.shape[0]
    N = w_int8.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    is_group = scale.dim() == 2
    # 投机解码 verify（M=1+N≤8）：双后端 int8 GEMM（权重 HBM 只读一次，shared 内
    # dequant→bf16 + GEMM），比 int8 GEMV 的 M× 权重读快 ~12x（mlp_gu 3.59ms→0.29ms）。
    # w8_linear 是 verify 路径所有 int8 线性（MLP dense_swiglu + attention _lin_prefill）
    # 的统一分派点，故在此路由。
    if is_group and M <= 8:
        from kernel.gemm_int8_triton import verify_gemm_enabled, verify_int8_gemm
        if verify_gemm_enabled():
            return verify_int8_gemm(x, w_int8, scale, out)
    # decode（M=bs，小）走 int8 GEMV（kernel 支持 M>1 via grid.y，权重带宽减半，且
    # 不产生反量化临时 buffer——27B int8 权重 27G 常驻，反量化临时 buffer 会 OOM）。
    # prefill（M 大）反量化后 matmul（tensor-core 跨 M 复用权重，比 int8 GEMV 的 M×
    # 权重读快；反量化临时 buffer 在 graph 外，4G headroom 够）。
    if M <= 32 and gemv_int8_available() and os.environ.get(env, "1") != "0":
        if is_group:
            _mod.gemv_int8_group(x, w_int8, scale, out)
        else:
            _mod.gemv_int8(x, w_int8, scale, out)
    else:
        # 反量化：per-channel scale [N]→[N,1]；group scale [N,K/128]→repeat_interleave 128
        if is_group:
            sc = scale.repeat_interleave(128, dim=1)
        else:
            sc = scale.unsqueeze(1)
        w_bf16 = (w_int8.float() * sc).to(x.dtype)
        torch.matmul(x, w_bf16.t(), out=out)
    return out
