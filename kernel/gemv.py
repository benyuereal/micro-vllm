"""手写 CUDA GEMV：单用户 decode（M=1）替代 cuBLAS，快 32-44%。

W 用 [N,K] 连续布局（prepare_weights 全局改存 [N,K]，零额外显存）。
M=1 全胜 cuBLAS；M>=4 cuBLAS 切 tensor-core GEMM 反超——故仅 M=1 调用此 kernel。

首次调用 cpp_extension.load_inline 编译（~30s），之后走 ~/.cache/torch_extensions 缓存。
编译失败自动 fallback torch.matmul（gemv_available()=False）。
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
        cu_path = os.path.join(os.path.dirname(__file__), "gemv.cu")
        with open(cu_path) as f:
            cuda_src = f.read()
        cpp_src = "torch::Tensor gemv_v2(torch::Tensor x, torch::Tensor w_t, torch::Tensor out);"
        _mod = load_inline(
            name="micro_gemv_v2",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["gemv_v2"],
            extra_cuda_cflags=["-O3", "-arch=sm_89"],
            verbose=False,
        )
        logger.info("gemv_v2 kernel 编译成功（手写 CUDA GEMV）")
    except Exception as e:
        _init_err = e
        logger.warning(f"gemv_v2 编译失败，fallback torch.matmul: {e}")


def gemv_available() -> bool:
    """手写 kernel 是否可用（编译成功）。"""
    if _mod is None and _init_err is None:
        _load()
    return _mod is not None


def gemv_v2(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """out = x @ w.t()，w 为 [N,K] 连续布局（prepare_weights 改存后）。

    Args:
        x:   [M, K] bf16 contiguous（M=1 最优）
        w:   [N, K] bf16 contiguous（[N,K] 布局，每输出行连续 K）
        out: [M, N] bf16 contiguous（可选，graph 友好的 in-place 写入）
    Returns:
        out: [M, N]
    """
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if not gemv_available():
        # fallback：w 是 [N,K]，x @ w.t() = [M,K]@[K,N]
        return torch.matmul(x, w.t(), out=out)
    _mod.gemv_v2(x, w, out)
    return out
