"""实验：routed gate_up 列 tile 大小对 HBM 带宽利用的影响。

gate_up 实测 93.1us/层，HBM floor 80.4us/层 = 1.16x（86% 带宽）。
gap 来源假设：(1) 波尾（132 block/92 SM=1.43 波）(2) num_stages=2 pipeline 深度不足。
增大列 tile BLOCK_I: 64→128→224 减少 block 数（132→66→33），消除波尾。
isolation 测绝对时间 + 带Utility，graph 路径差分法测真实收益。

注意 isolation 不可信（L2 热），但带宽利用率(实测时间 vs HBM floor)可看 tile 是否吃满带宽。
graph 路径才是决策依据。
"""
import sys, torch, tilelang, tilelang.language as T
sys.path.insert(0, "/models/micro-vllm")

H=2048; INTER=1408; K=6; E=64; N=1; TWO_INTER=2*INTER
accum = T.float32

def make_gate_up(BLOCK_I, num_stages=2):
    @tilelang.jit(out_idx=[4])
    def _k(N, H, INTER, E, K, dtype):
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
            with T.Kernel(N, K, T.ceildiv(INTER, BLOCK_I), threads=128) as (bn, kid, iblk):
                X_s = T.alloc_shared([16, 128], dtype)
                Wg_s = T.alloc_shared([BLOCK_I, 128], dtype)
                Wu_s = T.alloc_shared([BLOCK_I, 128], dtype)
                g_acc = T.alloc_fragment([16, BLOCK_I], accum)
                u_acc = T.alloc_fragment([16, BLOCK_I], accum)
                g_s = T.alloc_shared([16, BLOCK_I], accum)
                u_s = T.alloc_shared([16, BLOCK_I], accum)
                e = IDX[bn, kid]
                wk = W_gate[bn, kid]
                T.clear(g_acc); T.clear(u_acc)
                for kh in T.Pipelined(T.ceildiv(H, 128), num_stages=num_stages):
                    T.copy(X16[bn, 0:16, kh * 128:(kh + 1) * 128], X_s)
                    T.copy(W_gu[e, iblk * BLOCK_I:(iblk + 1) * BLOCK_I, kh * 128:(kh + 1) * 128], Wg_s)
                    T.copy(W_gu[e, INTER + iblk * BLOCK_I:INTER + (iblk + 1) * BLOCK_I, kh * 128:(kh + 1) * 128], Wu_s)
                    T.gemm(X_s, Wg_s, g_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(X_s, Wu_s, u_acc, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(g_acc, g_s); T.copy(u_acc, u_s)
                for j in T.Parallel(BLOCK_I):
                    if iblk * BLOCK_I + j < INTER:
                        g = g_s[0, j]
                        sig = 1.0 / (1.0 + T.exp(-g))
                        Act16[bn, kid, kid, iblk * BLOCK_I + j] = (g * sig * u_s[0, j] * wk).astype(dtype)
        return main
    return _k(N, H, INTER, E, K, T.bfloat16)


def bench_iso(k, BLOCK_I, label):
    torch.manual_seed(0)
    x16 = torch.zeros(N,16,H,dtype=torch.bfloat16,device='cuda'); x16[:,0,:]=torch.randn(N,H,dtype=torch.bfloat16,device='cuda')*0.1
    wgu = torch.randn(E,TWO_INTER,H,dtype=torch.bfloat16,device='cuda')*0.02
    idx = torch.zeros(N,K,dtype=torch.int32,device='cuda'); idx[0]=torch.randint(0,E,(K,))
    wg = torch.ones(N,K,dtype=torch.bfloat16,device='cuda')
    act = k(x16, wgu, idx, wg)
    grid = N * K * ((INTER + BLOCK_I - 1)//BLOCK_I)
    # warmup
    for _ in range(20): k(x16, wgu, idx, wg)
    torch.cuda.synchronize()
    ev0=torch.cuda.Event(enable_timing=True); ev1=torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(500): k(x16, wgu, idx, wg)
    ev1.record(); torch.cuda.synchronize()
    us = ev0.elapsed_time(ev1)/500*1000
    # HBM floor per call (单层单次)
    hbm = K*2*INTER*H*2 + K*16*INTER*2  # weight + write
    floor_us = hbm/864e9*1e6
    print(f"  {label:20s} BLOCK_I={BLOCK_I:3d} grid={grid:3d} iso={us:6.2f}us  floor={floor_us:5.1f}us  bw_util={floor_us/us*100:.0f}%")
    return us, act


if __name__ == "__main__":
    print("routed gate_up tile 实验 (isolation, L2 热, 只看带宽利用趋势):")
    acts = {}
    for bi, ns in [(64,2), (128,2), (224,2), (64,3), (128,3), (352,2)]:
        try:
            k = make_gate_up(bi, ns)
            us, act = bench_iso(k, bi, f"BI={bi},ns={ns}")
            acts[(bi,ns)] = act
        except Exception as e:
            print(f"  BLOCK_I={bi},ns={ns} FAIL: {str(e)[:150]}")
    # 正确性：以 BI=64,ns=2 为 reference 对比
    if (64,2) in acts and (128,2) in acts:
        ref = acts[(64,2)]; test = acts[(128,2)]
        # 不同 tile 切分，对比 kid 行的 act
        r = ref[0,:,0,:]; t = test[0,:,0,:]
        diff = (r-t).abs().max().item() / (r.abs().max().item()+1e-9)
        print(f"  correctness BI=128 vs BI=64: max_rel_diff={diff:.4f}")
