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
                   "torch::Tensor scale, torch::Tensor out);")
        _mod = load_inline(
            name="micro_gemv_int8",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gemv_int8"],
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
    """W8A16 线性层：x [M,K] bf16, w_int8 [N,K] int8, scale [N] fp32 → out [M,N] bf16。

    M=1 且 int8 kernel 可用且 env 开启 → 手写 int8 GEMV（权重带宽减半）；
    否则反量化 int8→bf16 后 x @ w.t()（prefill M>1 走此路，compute-bound 反量化开销可忽略）。
    """
    M = x.shape[0]
    N = w_int8.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if M == 1 and gemv_int8_available() and os.environ.get(env, "1") != "0":
        _mod.gemv_int8(x, w_int8, scale, out)
    else:
        w_bf16 = (w_int8.float() * scale.unsqueeze(1)).to(x.dtype)
        torch.matmul(x, w_bf16.t(), out=out)
    return out
