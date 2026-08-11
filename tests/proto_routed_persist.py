"""原型 v1：routed persistent kernel（gate_up + down 两阶段，gate 留独立 kernel 先验证）。

验证关键点：
1. T.Kernel(NUM_SMS) + T.serial(sm_idx, n_task, NUM_SMS) 吃任务
2. 全局 buffer act16 跨阶段通信（gate_up 写 → sync_grid → down 读）
3. T.gemm 在 persistent 循环体内能否编译
4. 与独立 2-kernel 对比正确性 + 性能

grid=(NUM_SMS=92,)。
阶段1: routed_gate_up，132 task（K=6 × cdiv(INTER,64)=22），每 task 算 (kid,iblk) 的 act
  → 写 act16[N,K,16,INTER]（全局）
  sync_grid
阶段2: routed_down，64 task（cdiv(H,32)），每 task 串行 K expert 累加 → out[N,H]
"""
import sys, torch
sys.path.insert(0, "/models/micro-vllm")
import tilelang
import tilelang.language as T

_TORCH_TO_TL = {torch.float16: T.float16, torch.bfloat16: T.bfloat16}
NUM_SMS = 92


@tilelang.jit(out_idx=[6])
def routed_persistent_kernel(N, H, INTER, E, K, dtype, BLOCK_H=32):
    # out_idx=[6]: Act16(idx5) 是中间全局 buffer，需作为输入传入；Out(idx6) 是唯一输出
    # 所以调用: ker(X16, Egu, Ed, IDX, WG, Act16) → 返回 Out
    """persistent: routed_gate_up(132 task) → sync_grid → routed_down(64 task)。
    输入: X16[N,16,H], Egu[E,2*INTER,H], Ed[E,H,INTER], IDX[N,K] int32, WG[N,K]
    中间(全局): Act16[N,K,16,INTER]
    输出: Out[N,H]
    """
    accum = T.float32
    TWO_INTER = 2 * INTER
    N_GU = K * T.ceildiv(INTER, 64)      # 132
    N_DN = T.ceildiv(H, BLOCK_H)          # 64
    threads = 128

    @T.prim_func
    def main(
        X16: T.Tensor([N, 16, H], dtype),
        Egu: T.Tensor([E, TWO_INTER, H], dtype),
        Ed: T.Tensor([E, H, INTER], dtype),
        IDX: T.Tensor([N, K], T.int32),
        WG: T.Tensor([N, K], dtype),
        Act16: T.Tensor([N, K, 16, INTER], dtype),
        Out: T.Tensor([N, H], dtype),
    ):
        with T.Kernel(NUM_SMS, threads=threads) as (sm_idx,):
            # ===== 阶段1: routed_gate_up =====
            for task in T.serial(sm_idx, N_GU, NUM_SMS):
                nblk = T.ceildiv(INTER, 64)
                kid = task // nblk
                iblk = task % nblk
                e = IDX[0, kid]
                wk = WG[0, kid]
                X_s = T.alloc_shared([16, 128], dtype)
                Wg_s = T.alloc_shared([64, 128], dtype)
                Wu_s = T.alloc_shared([64, 128], dtype)
                g_acc = T.alloc_fragment([16, 64], accum)
                u_acc = T.alloc_fragment([16, 64], accum)
                g_s = T.alloc_shared([16, 64], accum)
                u_s = T.alloc_shared([16, 64], accum)
                T.clear(g_acc); T.clear(u_acc)
                for kh in T.Pipelined(T.ceildiv(H, 128), num_stages=2):
                    T.copy(X16[0, 0:16, kh * 128:(kh + 1) * 128], X_s)
                    T.copy(Egu[e, iblk * 64:(iblk + 1) * 64, kh * 128:(kh + 1) * 128], Wg_s)
                    T.copy(Egu[e, INTER + iblk * 64:INTER + (iblk + 1) * 64,
                               kh * 128:(kh + 1) * 128], Wu_s)
                    T.gemm(X_s, Wg_s, g_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(X_s, Wu_s, u_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(g_acc, g_s); T.copy(u_acc, u_s)
                for j in T.Parallel(64):
                    g = g_s[0, j]
                    sig = 1.0 / (1.0 + T.exp(-g))
                    Act16[0, kid, kid, iblk * 64 + j] = (g * sig * u_s[0, j] * wk).astype(dtype)
            T.sync_grid()
            # ===== 阶段2: routed_down =====
            for task in T.serial(sm_idx, N_DN, NUM_SMS):
                hblk = task
                A_s = T.alloc_shared([16, 128], dtype)
                W_s = T.alloc_shared([BLOCK_H, 128], dtype)
                acc = T.alloc_fragment([16, BLOCK_H], accum)
                acc_s = T.alloc_shared([16, BLOCK_H], accum)
                out_frag = T.alloc_fragment([BLOCK_H], accum)
                T.clear(out_frag)
                for k in T.serial(K):
                    e = IDX[0, k]
                    T.clear(acc)
                    for ki in T.Pipelined(T.ceildiv(INTER, 128), num_stages=2):
                        T.copy(Act16[0, k, 0:16, ki * 128:(ki + 1) * 128], A_s)
                        T.copy(Ed[e, hblk * BLOCK_H:(hblk + 1) * BLOCK_H, ki * 128:(ki + 1) * 128], W_s)
                        T.gemm(A_s, W_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.copy(acc, acc_s)
                    for j in T.Parallel(BLOCK_H):
                        out_frag[j] += acc_s[k, j]
                for j in T.Parallel(BLOCK_H):
                    Out[0, hblk * BLOCK_H + j] = out_frag[j].astype(dtype)
    return main


def test():
    import torch.nn.functional as F
    from kernel.moe import moe_gate_up_kernel, moe_down_kernel, _TORCH_TO_TL as TTL
    torch.manual_seed(42)
    N, H, INTER, E, K = 1, 2048, 1408, 64, 6
    dtype = torch.bfloat16; tl_dt = _TORCH_TO_TL[dtype]
    x16 = torch.randn(N, 16, H, device="cuda", dtype=dtype) * 0.1
    x16[:, 1:, :] = 0  # 只 row 0 真实
    e_gu = torch.randn(E, 2*INTER, H, device="cuda", dtype=dtype) * 0.02
    e_d = torch.randn(E, H, INTER, device="cuda", dtype=dtype) * 0.02
    idx = torch.randint(0, E, (N, K), device="cuda", dtype=torch.int32)
    wg = torch.rand(N, K, device="cuda", dtype=dtype)

    # 独立 2-kernel 参考
    k_gu = moe_gate_up_kernel(N, H, INTER, E, K, tl_dt)
    k_dn = moe_down_kernel(N, H, INTER, E, K, tl_dt)
    act_ref = k_gu(x16, e_gu, idx, wg)
    out_ref = k_dn(act_ref, e_d, idx)

    # persistent kernel（Act16 需预分配作为输入传入）
    ker = routed_persistent_kernel(N, H, INTER, E, K, tl_dt)
    act16_buf = torch.zeros(N, K, 16, INTER, device="cuda", dtype=dtype)
    out_k = ker(x16, e_gu, e_d, idx, wg, act16_buf)

    print(f"out_ref shape={out_ref.shape}, out_k shape={out_k.shape}")
    d = (out_ref.float() - out_k.float()).abs()
    print(f"maxdiff={d.max().item():.6f}, mean={d.mean().item():.6f}")
    print(f"ref[:8]={out_ref[0,:8].tolist()}")
    print(f"ker[:8]={out_k[0,:8].tolist()}")

    # 性能
    def t(fn, iters=300):
        for _ in range(30): fn()
        torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)/iters*1000
    print(f"\n--- isolation 性能（参考，graph 下可能不同）---")
    print(f"独立 gate_up+down: {t(lambda: k_dn(k_gu(x16, e_gu, idx, wg), e_d, idx)):.1f} us")
    print(f"persistent:        {t(lambda: ker(x16, e_gu, e_d, idx, wg, act16_buf)):.1f} us")


if __name__ == "__main__":
    test()
