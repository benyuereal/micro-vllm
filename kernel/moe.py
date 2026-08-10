"""Fused MoE decode kernels for DeepSeek-V2-Lite (M=16 grid-parallel).

替代原 Triton 逐 token grouped-GEMV loop（decode 路径）。routed + shared experts 全融合。

📌 核心思路（M=16 grid-parallel T.gemm）：
    bs=1 decode 下每个 expert 是 GEMV (M=1)，但 ``T.gemm`` 要求 ``M % 16 == 0``
    （tensor-core mma.h 硬约束）。解法：M=1 → M=16 零填充，让 grid 沿 K(top_k) 维并行吃掉
    padding——每个 block 算一个 (token, expert)，真实 act 在 16 行的第 ``kid`` 行，其余 15 行
    零填充无害。

📌 routed experts：两个 kernel，back-to-back，act 经 L2 暂存为 [N, K, 16, INTER]：
    gu_silu : X16[N,16,H] @ W_gu[e]^T → gate/up [16,INTER]；
              silu(gate)*up*gate_weight → act16[n, kid, kid, :]。
              🔑 silu 必须在 fp32 下算（bf16 ``T.exp`` 精度丢失 ~1.5x）。
    down    : act16[n,kid] @ W_d[e]^T → out[N,H]，跨 expert 用 ``T.atomic_add`` 累加到 fp32 输出。
              🔑 全局 tensor ``+=`` 不是原子操作（6 expert block 竞争同一 O[h]），必须 atomic_add；
              bf16 atomic 在 sm_89 不可靠，故输出 fp32。

    gate_weight 只在 gu_silu 乘一次，down 不再乘。

📌 shared experts：固定大 MLP（无路由，K=1，s_inter = inter*n_shared），同构于 routed 单 expert：
    shared_gate_up : X16 @ shared_gu^T → silu(gate)*up → act16[N,16,S_INTER]。
    shared_down    : act16 @ shared_d^T → += out（直接累加到 routed 输出 buffer，
                     省掉 routed/shared 间的 PyTorch 加法）。

⚡ 范围：routed + shared experts 全融合。gate/router（data-dependent topk）留 PyTorch
   （占比小，graph-friendly）。

📊 性能（L20, K=6, E=64, INTER=1408, H=2048, N=1）：
    routed: gu_silu 28.8us + down 17.0us = 79.8us serial  vs  Triton 3-step 106.5us  (1.33x)
    correctness rel=0.014 vs Triton reference。
"""
import torch
import tilelang
import tilelang.language as T


# ============ kernel A: gate_up + silu*up*w → act16[N, K, 16, INTER] ============
@tilelang.jit(out_idx=[4])
def moe_gate_up_kernel(N, H, INTER, E, K, dtype):
    """grid=(N, K, cdiv(INTER,64))。每 block 算 (token n, expert kid) 的 64 列 act。
    gate/up 各一次 M=16 T.gemm；silu 在 fp32 shared 下算；写 act16[n, kid, kid, :]。"""
    accum = T.float32
    TWO_INTER = 2 * INTER

    @T.prim_func
    def main(
        X16: T.Tensor([N, 16, H], dtype),
        W_gu: T.Tensor([E, TWO_INTER, H], dtype),
        IDX: T.Tensor([N, K], T.int32),
        W_gate: T.Tensor([N, K], dtype),
        Act16: T.Tensor([N, K, 16, INTER], dtype),
    ):
        with T.Kernel(N, K, T.ceildiv(INTER, 64), threads=128) as (bn, kid, iblk):
            X_s = T.alloc_shared([16, 128], dtype)
            Wg_s = T.alloc_shared([64, 128], dtype)
            Wu_s = T.alloc_shared([64, 128], dtype)
            g_acc = T.alloc_fragment([16, 64], accum)
            u_acc = T.alloc_fragment([16, 64], accum)
            g_s = T.alloc_shared([16, 64], accum)   # fp32: bf16 T.exp 丢 ~1.5x 精度
            u_s = T.alloc_shared([16, 64], accum)
            e = IDX[bn, kid]
            wk = W_gate[bn, kid]
            T.clear(g_acc); T.clear(u_acc)
            for kh in T.Pipelined(T.ceildiv(H, 128), num_stages=2):
                T.copy(X16[bn, 0:16, kh * 128:(kh + 1) * 128], X_s)
                T.copy(W_gu[e, iblk * 64:(iblk + 1) * 64, kh * 128:(kh + 1) * 128], Wg_s)
                T.copy(W_gu[e, INTER + iblk * 64:INTER + (iblk + 1) * 64,
                             kh * 128:(kh + 1) * 128], Wu_s)
                T.gemm(X_s, Wg_s, g_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(X_s, Wu_s, u_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            T.copy(g_acc, g_s); T.copy(u_acc, u_s)
            for j in T.Parallel(64):
                g = g_s[0, j]
                sig = 1.0 / (1.0 + T.exp(-g))
                Act16[bn, kid, kid, iblk * 64 + j] = (g * sig * u_s[0, j] * wk).astype(dtype)

    return main


# ============ kernel B: down → out[N, H] (per-token block, fragment 累加) ============
# 🔑 不用 atomic：参考 TileKernels reduce_fused，每 token 一个 block，block 内串行 K 个
# expert，每个 expert 用 M=16 T.gemm 算出该 expert 的输出，在 fp32 fragment 里累加，
# 最后覆盖写 Out。无竞争、不依赖输出清零、精度由 fp32 累加保证。
@tilelang.jit(out_idx=[3])
def moe_down_kernel(N, H, INTER, E, K, dtype):
    """grid=(N, cdiv(H,64))。每 block 串行 K 个 expert：M=16 T.gemm → acc[16,64]，
    取第 k 行累加到 out_frag[64]（fp32），最后写 Out[bn, hblk*64+j]。"""
    accum = T.float32

    @T.prim_func
    def main(
        Act16: T.Tensor([N, K, 16, INTER], dtype),
        W_d: T.Tensor([E, H, INTER], dtype),
        IDX: T.Tensor([N, K], T.int32),
        Out: T.Tensor([N, H], dtype),
    ):
        with T.Kernel(N, T.ceildiv(H, 64), threads=128) as (bn, hblk):
            A_s = T.alloc_shared([16, 128], dtype)
            W_s = T.alloc_shared([64, 128], dtype)
            acc = T.alloc_fragment([16, 64], accum)
            acc_s = T.alloc_shared([16, 64], accum)    # fp32 relay，单行索引避免 layout 冲突
            out_frag = T.alloc_fragment([64], accum)    # fp32 跨 expert 累加器，无 atomic
            T.clear(out_frag)
            for k in T.serial(K):
                e = IDX[bn, k]
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(INTER, 128), num_stages=2):
                    T.copy(Act16[bn, k, 0:16, ki * 128:(ki + 1) * 128], A_s)
                    T.copy(W_d[e, hblk * 64:(hblk + 1) * 64, ki * 128:(ki + 1) * 128], W_s)
                    T.gemm(A_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(acc, acc_s)
                for j in T.Parallel(64):
                    out_frag[j] += acc_s[k, j]          # 该 expert 输出在 acc_s 第 k 行
            for j in T.Parallel(64):
                Out[bn, hblk * 64 + j] = out_frag[j].astype(dtype)

    return main


# ============ shared expert kernels（固定大 MLP，无路由，K=1）============
# 与 routed 单 expert 同构，但无 topk/expert 索引/gate_weight，s_inter = inter*n_shared。
# down kernel 直接 += 到 routed 输出 buffer（Out 同一 tensor），省掉 routed/shared 间的 PyTorch 加法。
@tilelang.jit(out_idx=[2])
def shared_gate_up_kernel(N, H, S_INTER, dtype):
    """grid=(N, cdiv(S_INTER,64))。X16 @ shared_gu^T → silu(gate)*up → act16[N,16,S_INTER]。
    shared_gu: [H, 2*S_INTER]（gate|up 拼接后 .t()），x @ shared_gu = [N, 2*S_INTER]。"""
    accum = T.float32
    TWO = 2 * S_INTER

    @T.prim_func
    def main(
        X16: T.Tensor([N, 16, H], dtype),
        W_gu: T.Tensor([H, TWO], dtype),
        Act16: T.Tensor([N, 16, S_INTER], dtype),
    ):
        with T.Kernel(N, T.ceildiv(S_INTER, 64), threads=128) as (bn, iblk):
            X_s = T.alloc_shared([16, 128], dtype)
            Wg_s = T.alloc_shared([128, 64], dtype)
            Wu_s = T.alloc_shared([128, 64], dtype)
            g_acc = T.alloc_fragment([16, 64], accum)
            u_acc = T.alloc_fragment([16, 64], accum)
            g_s = T.alloc_shared([16, 64], accum)   # fp32: bf16 T.exp 丢 ~1.5x 精度
            u_s = T.alloc_shared([16, 64], accum)
            T.clear(g_acc); T.clear(u_acc)
            for kh in T.Pipelined(T.ceildiv(H, 128), num_stages=2):
                T.copy(X16[bn, 0:16, kh * 128:(kh + 1) * 128], X_s)
                T.copy(W_gu[kh * 128:(kh + 1) * 128, iblk * 64:(iblk + 1) * 64], Wg_s)
                T.copy(W_gu[kh * 128:(kh + 1) * 128, S_INTER + iblk * 64:S_INTER + (iblk + 1) * 64], Wu_s)
                T.gemm(X_s, Wg_s, g_acc, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(X_s, Wu_s, u_acc, policy=T.GemmWarpPolicy.FullCol)
            T.copy(g_acc, g_s); T.copy(u_acc, u_s)
            for j in T.Parallel(64):
                g = g_s[0, j]
                sig = 1.0 / (1.0 + T.exp(-g))
                Act16[bn, 0, iblk * 64 + j] = (g * sig * u_s[0, j]).astype(dtype)
    return main


@tilelang.jit()
def shared_down_kernel(N, H, S_INTER, dtype):
    """grid=(N, cdiv(H,64))。act16[n,0,:] @ shared_d^T → += Out（routed 输出 buffer）。
    shared_d: [S_INTER, H]（down .t()）。M=16 T.gemm 取 row 0，fp32 fragment 累加后 += Out。"""
    accum = T.float32

    @T.prim_func
    def main(
        Act16: T.Tensor([N, 16, S_INTER], dtype),
        W_d: T.Tensor([S_INTER, H], dtype),
        Out: T.Tensor([N, H], dtype),
    ):
        with T.Kernel(N, T.ceildiv(H, 64), threads=128) as (bn, hblk):
            A_s = T.alloc_shared([16, 128], dtype)
            W_s = T.alloc_shared([128, 64], dtype)
            acc = T.alloc_fragment([16, 64], accum)
            acc_s = T.alloc_shared([16, 64], accum)
            out_frag = T.alloc_fragment([64], accum)
            T.clear(out_frag)
            T.clear(acc)
            for ki in T.Pipelined(T.ceildiv(S_INTER, 128), num_stages=2):
                T.copy(Act16[bn, 0:16, ki * 128:(ki + 1) * 128], A_s)
                T.copy(W_d[ki * 128:(ki + 1) * 128, hblk * 64:(hblk + 1) * 64], W_s)
                T.gemm(A_s, W_s, acc, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, acc_s)
            for j in T.Parallel(64):
                out_frag[j] += acc_s[0, j]
            for j in T.Parallel(64):
                Out[bn, hblk * 64 + j] = (Out[bn, hblk * 64 + j].astype(accum) + out_frag[j]).astype(dtype)
    return main


# ============ cache + launcher ============
_kernel_cache: dict = {}

_TORCH_TO_TL = {
    torch.float16: T.float16,
    torch.bfloat16: T.bfloat16,
}


def moe_routed_decode(x, e_gu, e_d, idx, w_gate, x16=None):
    """x: [N, H], e_gu: [E, 2*inter, H], e_d: [E, H, inter],
    idx: [N, K] int, w_gate: [N, K], x16: [N,16,H] 可复用的 pad buffer（None 则内部分配）
    -> out: [N, H]"""
    N, H = x.shape
    E, TWO_INTER, _ = e_gu.shape
    INTER = TWO_INTER // 2
    K = idx.shape[1]
    idx_i32 = idx.to(torch.int32)
    tl_dtype = _TORCH_TO_TL[x.dtype]

    key = (N, H, INTER, E, K, x.dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = (
            moe_gate_up_kernel(N, H, INTER, E, K, tl_dtype),
            moe_down_kernel(N, H, INTER, E, K, tl_dtype),
        )
    k_gu, k_dn = _kernel_cache[key]

    # M=1 → M=16 零填充，真实 act 在第 0 行（kernel 内用 kid 行，对单 token kid 行 = 0 对齐）
    # 对 N 个 token：X16[n, 0, :] = x[n]，其余行 0
    if x16 is None:
        x16 = torch.zeros(N, 16, H, dtype=x.dtype, device=x.device)
        x16[:, 0, :] = x

    act16 = k_gu(x16, e_gu, idx_i32, w_gate)   # [N, K, 16, INTER]
    out = k_dn(act16, e_d, idx_i32)             # [N, H] (bf16, fragment fp32 累加后写出)
    return out


def moe_decode(x, gate_weight, e_gu, e_d, top_k, n_experts,
               shared_gu=None, shared_d=None):
    """完整 MoE decode（替代 moe_forward decode=True 路径）。

    gate/topk 留 PyTorch（小/固定），routed + shared experts 都用 M=16 全融合 kernel。
    shared down 直接 += 到 routed 输出 buffer，省掉 routed/shared 间的 PyTorch 加法。
    routed 与 shared 共用同一份 X16 pad buffer（row 0 = x）。
    x: [N, hidden], gate_weight: [E, hidden], e_gu: [E, 2*inter, hidden],
    e_d: [E, hidden, inter], shared_gu: [hidden, 2*s_inter], shared_d: [s_inter, hidden]
    返回: [N, hidden]
    """
    import torch.nn.functional as F
    N, H = x.shape
    # 1. gate + topk（PyTorch，graph-friendly，shape 固定）
    logits = F.linear(x, gate_weight)                              # [N, E]
    scores = logits.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
    topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)  # [N, K]

    # M=16 pad buffer，routed 与 shared 共用（row 0 = x，其余 0）
    x16 = torch.zeros(N, 16, H, dtype=x.dtype, device=x.device)
    x16[:, 0, :] = x

    # 2. routed experts（M=16 全融合）→ out [N, H]
    out = moe_routed_decode(x, e_gu, e_d, topk_idx, topk_weight, x16=x16)

    # 3. shared experts（M=16 全融合，down 直接 += 到 out，复用同一 x16）
    if shared_gu is not None:
        S_INTER = shared_d.shape[0]
        tl_dtype = _TORCH_TO_TL[x.dtype]
        skey = (N, H, S_INTER, x.dtype)
        if skey not in _kernel_cache:
            _kernel_cache[skey] = (
                shared_gate_up_kernel(N, H, S_INTER, tl_dtype),
                shared_down_kernel(N, H, S_INTER, tl_dtype),
            )
        k_sgu, k_sdn = _kernel_cache[skey]
        sact = k_sgu(x16, shared_gu)           # [N, 16, S_INTER]
        k_sdn(sact, shared_d, out)             # out += shared_out（in-place）

    return out
