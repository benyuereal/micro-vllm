"""TileLang MoE decode kernel：routed experts 全融合（gate_up + silu*up*w + down）。

替代 kernel/grouped_gemv.py 的 Triton 逐 token loop（16 次 kernel launch, 1055us/层）。

📌 数据流（单 token, K=top_k experts）：
    1. gate_up[k, 2*inter] = x[hidden] @ W_gu[idx[k], 2*inter, hidden].T   # K 次 GEMV
    2. act[k, inter] = silu(gate_up[k, :inter]) * gate_up[k, inter:] * w[k]
    3. out[hidden] += sum_k act[k, inter] @ W_d[idx[k], hidden, inter].T     # K 次加权 GEMV 累加

🔑 优化点（对应 TileRT execution gap）：
    - 16 次 kernel launch → 2 次 TileLang kernel（gate_up + down）
    - act[N, K, inter] 落 HBM 但小（N=8: 8×6×1408×2B=135KB），L2 命中，非 round-trip
    - expert_idx 间接寻址：e = IDX[bn]; T.copy(W[e, ...], W_shared)
    - gate_up kernel 每 token 一个 block，不重复计算；down kernel 按输出列并行

⚡ 范围：仅 routed experts。gate/router（data-dependent topk）和 shared experts 留在 PyTorch
   （gate/topk 42us/层仅占 3.6%，shared 75us 是固定大 GEMM，下阶段融合）。
"""
import torch
import tilelang
import tilelang.language as T


# ============ kernel A: gate_up + silu*up*w → act[N, K, INTER] ============
@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def moe_gate_up_kernel(N, H, INTER, E, K, BLOCK_H, BLOCK_INTER, dtype, num_stages=2):
    """每 block 一个 token，算该 token 的 K 个 expert 的 act[K, INTER]。"""
    accum_dtype = T.float32
    TWO_INTER = 2 * INTER

    @T.prim_func
    def main(
        X: T.Tensor([N, H], dtype),
        W_gu: T.Tensor([E, TWO_INTER, H], dtype),
        IDX: T.Tensor([N, K], T.int32),
        W_gate: T.Tensor([N, K], dtype),
        Act: T.Tensor([N, K, INTER], dtype),
    ):
        with T.Kernel(N, threads=256) as (bn,):
            X_shared = T.alloc_shared([BLOCK_H], dtype)
            Wg_shared = T.alloc_shared([BLOCK_INTER, BLOCK_H], dtype)
            Wu_shared = T.alloc_shared([BLOCK_INTER, BLOCK_H], dtype)
            gate_acc = T.alloc_fragment([BLOCK_INTER], accum_dtype)
            up_acc = T.alloc_fragment([BLOCK_INTER], accum_dtype)
            prod_v = T.alloc_fragment([BLOCK_INTER, BLOCK_H], accum_dtype)
            wk = T.alloc_local([1], dtype)

            for k in T.serial(K):
                e = IDX[bn, k]
                wk[0] = W_gate[bn, k]
                for bi in T.serial(T.ceildiv(INTER, BLOCK_INTER)):
                    T.fill(gate_acc, 0)
                    T.fill(up_acc, 0)
                    for kh in T.Pipelined(T.ceildiv(H, BLOCK_H), num_stages=num_stages):
                        T.copy(X[bn, kh * BLOCK_H:(kh + 1) * BLOCK_H], X_shared)
                        T.copy(
                            W_gu[e, bi * BLOCK_INTER:(bi + 1) * BLOCK_INTER, kh * BLOCK_H:(kh + 1) * BLOCK_H],
                            Wg_shared,
                        )
                        T.copy(
                            W_gu[e, INTER + bi * BLOCK_INTER:INTER + (bi + 1) * BLOCK_INTER, kh * BLOCK_H:(kh + 1) * BLOCK_H],
                            Wu_shared,
                        )
                        for i, j in T.Parallel(BLOCK_INTER, BLOCK_H):
                            prod_v[i, j] = Wg_shared[i, j].astype(accum_dtype) * X_shared[j].astype(accum_dtype)
                        T.reduce_sum(prod_v, gate_acc, dim=1, clear=False)
                        for i, j in T.Parallel(BLOCK_INTER, BLOCK_H):
                            prod_v[i, j] = Wu_shared[i, j].astype(accum_dtype) * X_shared[j].astype(accum_dtype)
                        T.reduce_sum(prod_v, up_acc, dim=1, clear=False)
                    for i in T.Parallel(BLOCK_INTER):
                        g = gate_acc[i]
                        sig = 1.0 / (1.0 + T.exp(-g))
                        Act[bn, k, bi * BLOCK_INTER + i] = (g * sig * up_acc[i] * wk[0].astype(accum_dtype)).astype(dtype)

    return main


# ============ kernel B: down GEMV → out[N, H] ============
@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def moe_down_kernel(N, H, INTER, E, K, BLOCK_INTER, BLOCK_OUT, dtype, num_stages=2):
    """每 block 一个 (token, 输出列段)，sum_k act[k] @ W_d[idx[k]].T。"""
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Act: T.Tensor([N, K, INTER], dtype),
        W_d: T.Tensor([E, H, INTER], dtype),
        IDX: T.Tensor([N, K], T.int32),
        W_gate: T.Tensor([N, K], dtype),
        Out: T.Tensor([N, H], dtype),
    ):
        with T.Kernel(N, T.ceildiv(H, BLOCK_OUT), threads=256) as (bn, bo):
            Act_shared = T.alloc_shared([BLOCK_INTER], dtype)
            Wd_shared = T.alloc_shared([BLOCK_OUT, BLOCK_INTER], dtype)
            out_acc = T.alloc_fragment([BLOCK_OUT], accum_dtype)
            prod_d = T.alloc_fragment([BLOCK_OUT, BLOCK_INTER], accum_dtype)

            T.fill(out_acc, 0)
            for k in T.serial(K):
                e = IDX[bn, k]
                for ki in T.Pipelined(T.ceildiv(INTER, BLOCK_INTER), num_stages=num_stages):
                    T.copy(Act[bn, k, ki * BLOCK_INTER:(ki + 1) * BLOCK_INTER], Act_shared)
                    T.copy(
                        W_d[e, bo * BLOCK_OUT:(bo + 1) * BLOCK_OUT, ki * BLOCK_INTER:(ki + 1) * BLOCK_INTER],
                        Wd_shared,
                    )
                    for i, j in T.Parallel(BLOCK_OUT, BLOCK_INTER):
                        prod_d[i, j] = Wd_shared[i, j].astype(accum_dtype) * Act_shared[j].astype(accum_dtype)
                    T.reduce_sum(prod_d, out_acc, dim=1, clear=False)
            T.copy(out_acc, Out[bn, bo * BLOCK_OUT:(bo + 1) * BLOCK_OUT])

    return main


# ============ cache + launcher ============
_kernel_cache: dict = {}

_TORCH_TO_TL = {
    torch.float16: T.float16,
    torch.bfloat16: T.bfloat16,
}


def moe_routed_decode(x, e_gu, e_d, idx, w_gate):
    """x: [N, H], e_gu: [E, 2*inter, H], e_d: [E, H, inter], idx: [N, K], w_gate: [N, K] -> [N, H]"""
    N, H = x.shape
    E, TWO_INTER, _ = e_gu.shape
    INTER = TWO_INTER // 2
    K = idx.shape[1]
    idx_i32 = idx.to(torch.int32)
    tl_dtype = _TORCH_TO_TL[x.dtype]

    key = (N, H, INTER, E, K, x.dtype)
    if key not in _kernel_cache:
        # smem: L20 max dynamic=100KB。Pipelined num_stages=2 会 double-buffer shared。
        # gate_up: Wg/Wu [BI,BH]×2stages; BI=64,BH=128 → 2×64×128×2×2=64KB ✓
        # down: Wd [BO,BI]×2; BI=64,BO=128 → 128×64×2×2=32KB ✓
        _kernel_cache[key] = (
            moe_gate_up_kernel(N, H, INTER, E, K, BLOCK_H=128, BLOCK_INTER=64, dtype=tl_dtype, num_stages=2),
            moe_down_kernel(N, H, INTER, E, K, BLOCK_INTER=64, BLOCK_OUT=128, dtype=tl_dtype, num_stages=2),
        )
    k_gu, k_dn = _kernel_cache[key]
    act = k_gu(x, e_gu, idx_i32, w_gate)   # [N, K, INTER]
    out = k_dn(act, e_d, idx_i32, w_gate)  # [N, H]
    return out


def moe_decode_tilelang(x, gate_weight, e_gu, e_d, top_k, n_experts,
                        shared_gu=None, shared_d=None):
    """完整 MoE decode（替代 moe_forward decode=True 路径）。

    gate/topk/shared 留 PyTorch（小/固定），routed experts 用 TileLang 全融合。
    x: [N, hidden], gate_weight: [E, hidden], e_gu: [E, 2*inter, hidden],
    e_d: [E, hidden, inter], shared_gu: [hidden, 2*s_inter], shared_d: [s_inter, hidden]
    返回: [N, hidden]
    """
    import torch.nn.functional as F
    N = x.shape[0]
    # 1. gate + topk（PyTorch，graph-friendly，shape 固定）
    logits = F.linear(x, gate_weight)                              # [N, E]
    scores = logits.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
    topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)  # [N, K]

    # 2. routed experts（TileLang 全融合）
    out = moe_routed_decode(x, e_gu, e_d, topk_idx, topk_weight)  # [N, H]

    # 3. shared experts（PyTorch 大 GEMM，固定无路由）
    if shared_gu is not None:
        gate_up = x @ shared_gu                       # [N, 2*s_inter]
        gate, up = gate_up.chunk(2, dim=-1)
        out = out + (F.silu(gate) * up) @ shared_d    # [N, H]

    return out

