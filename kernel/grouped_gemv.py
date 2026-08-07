"""
Grouped GEMV for DeepSeek-V2-Lite MoE decode.

🎯 目的：
    单 token decode 时，top-k 个 expert 各做一次 SwiGLU（gate_up GEMV + silu*up + down GEMV）。
    PyTorch 路径需要先 gather 选中 expert 的权重（advanced index 拷贝 ~96MB，~700μs），
    再 bmm/loop 计算。本 kernel 在 Triton 内部按 expert_idx 索引权重行，**不显式 gather**，
    把 routed expert 计算从 ~520μs 降到接近纯算力下限。

🔑 数据流（单 token, K=top_k experts）：
    gate_up[k, j] = sum_h x[h] * W_gu[expert_idx[k], j, h]     # j ∈ [0, 2*inter)
    act[k, i]     = silu(gate_up[k, i]) * gate_up[k, i+inter]  # i ∈ [0, inter)
    out[h]       += sum_i (act[k, i] * w[k]) * W_d[expert_idx[k], h, i]

实现：
    - grouped_gate_up_kernel: grid=(K, cdiv(2*inter, BLOCK_N)), 输出 [K, 2*inter]
    - grouped_down_kernel:    grid=(cdiv(H, BLOCK_N),), 输出 [H]，K 段加权累加
    两者均按 expert_idx 在 [E, ...] 权重上索引，无权重拷贝。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gate_up_kernel(
    x_ptr, w_ptr, idx_ptr, out_ptr,
    M, OUT, K_dim, H,
    stride_xh,
    stride_we, stride_wo, stride_wh,
    stride_outk, stride_outo,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[k, n_block] = sum_h x[h] * W[idx[k], n_block, h];  M=1 (单 token).
    BLOCK_K 必须 >= H（单 token 的 H 维一次 load，不循环 K，避免循环边界 bug）。"""
    pid_k = tl.program_id(0)          # expert slot in [0, K)
    pid_n = tl.program_id(1)          # output 列块

    e = tl.load(idx_ptr + pid_k).to(tl.int64)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_n = rn < OUT
    mask_k = rk < H

    x = tl.load(x_ptr + rk * stride_xh, mask=mask_k, other=0.0)     # [BLOCK_K]
    w = tl.load(w_ptr + e * stride_we + rn[:, None] * stride_wo + rk[None, :] * stride_wh,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_N, BLOCK_K]
    acc = tl.sum(x[None, :] * w, axis=1)                            # [BLOCK_N]

    out_ptrs = out_ptr + pid_k * stride_outk + rn * stride_outo
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _grouped_down_kernel(
    act_ptr, w_ptr, idx_ptr, weight_ptr, out_ptr,
    INTER, K_dim, H,
    stride_actk, stride_acti,
    stride_we, stride_wh, stride_wi,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """out[h] = sum_k weight[k] * sum_i act[k,i] * W[idx[k], h, i];  单 token, 加权累加到 out[H].
    BLOCK_K 必须 >= INTER（inter 维一次 load）。"""
    pid_h = tl.program_id(0)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rk = tl.arange(0, BLOCK_K)
    mask_h = rh < H
    mask_i = rk < INTER

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in range(K_dim):
        e = tl.load(idx_ptr + k).to(tl.int64)
        wk = tl.load(weight_ptr + k)

        a = tl.load(act_ptr + k * stride_actk + rk * stride_acti, mask=mask_i, other=0.0)  # [BLOCK_K]
        w = tl.load(w_ptr + e * stride_we + rh[:, None] * stride_wh + rk[None, :] * stride_wi,
                    mask=mask_h[:, None] & mask_i[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]
        inner = tl.sum(w * a[None, :], axis=1)                      # [BLOCK_H]
        acc += inner * wk

    out_ptrs = out_ptr + rh
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_h)


def grouped_gate_up(x: torch.Tensor, w_gu: torch.Tensor, expert_idx: torch.Tensor,
                    out: torch.Tensor = None) -> torch.Tensor:
    """x: [1, H], w_gu: [E, 2*inter, H], expert_idx: [K] -> [K, 2*inter]."""
    _, H = x.shape
    E, OUT, _ = w_gu.shape
    K = expert_idx.shape[0]
    if out is None:
        out = torch.empty(K, OUT, dtype=x.dtype, device=x.device)
    BLOCK_N = 64
    # BLOCK_K 取 >= H 的 2 的幂，单 token H 维一次 load
    BLOCK_K = triton.next_power_of_2(H)
    grid = (K, triton.cdiv(OUT, BLOCK_N))
    _grouped_gate_up_kernel[grid](
        x, w_gu, expert_idx, out,
        1, OUT, K, H,
        x.stride(1),
        w_gu.stride(0), w_gu.stride(1), w_gu.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def grouped_down(act: torch.Tensor, w_d: torch.Tensor, expert_idx: torch.Tensor,
                 weight: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """act: [K, inter], w_d: [E, H, inter], expert_idx: [K], weight: [K] -> [1, H]."""
    K, INTER = act.shape
    E, H, _ = w_d.shape
    if out is None:
        out = torch.empty(1, H, dtype=act.dtype, device=act.device)
    BLOCK_H = 64
    BLOCK_K = triton.next_power_of_2(INTER)
    grid = (triton.cdiv(H, BLOCK_H),)
    _grouped_down_kernel[grid](
        act, w_d, expert_idx, weight, out,
        INTER, K, H,
        act.stride(0), act.stride(1),
        w_d.stride(0), w_d.stride(1), w_d.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
    )
    return out
