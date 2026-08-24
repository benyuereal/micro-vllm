"""隔离：triton recurrent kernel state vs 纯 torch（同输入），并打印 kv_mem 中间量。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch, triton
import triton.language as tl
from models.qwen3_5.adapter import _gdn_recurrent_prefill_kernel


@triton.jit
def kv_only_kernel(QKV, G, BETA, KVOUT, CU, SEQ_IDX,
                   H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   N_GDN, GDN_L, SCALE, BLOCK_D: tl.constexpr):
    # 只算最后一个 token 的 kv_mem（用 k[None,:] 版本，即当前 kernel 的写法）
    s = tl.program_id(0)
    h = tl.program_id(1)
    start = tl.load(CU + s)
    end = tl.load(CU + s + 1)
    L = end - start
    dk = tl.arange(0, BLOCK_D)
    dv = tl.arange(0, BLOCK_D)
    S_m = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    for i in range(0, L):
        t = start + i
        q_base = QKV + t.to(tl.int64) * (2 * H * DK + H * DV) + h * DK
        k_base = q_base + H * DK
        v_base = k_base + H * DK
        q = tl.load(q_base + dk).to(tl.float32)
        k = tl.load(k_base + dk).to(tl.float32)
        v = tl.load(v_base + dv).to(tl.float32)
        g = tl.load(G + t.to(tl.int64) * H + h).to(tl.float32)
        beta = tl.load(BETA + t.to(tl.int64) * H + h).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        ge = tl.exp(g)
        S_m = S_m * ge
        kv_mem = tl.sum(S_m * k[None, :], axis=0)
        delta = (v - kv_mem) * beta
        S_m += k[:, None] * delta[None, :]
    # 存最后一个 token 的 kv_mem（用当前 S_m 和最后 k 重算）
    t = start + L - 1
    q_base = QKV + t.to(tl.int64) * (2 * H * DK + H * DV) + h * DK
    k_base = q_base + H * DK
    k = tl.load(k_base + dk).to(tl.float32)
    k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    kv_mem = tl.sum(S_m * k[None, :], axis=0)
    tl.store(KVOUT + h * DV + dv, kv_mem)


def main():
    DK = DV = 128
    L = 8
    H = 16
    SCALE = DK ** -0.5
    dev = "cuda"
    torch.manual_seed(0)
    qkv = (torch.randn(L, 2 * H * DK + H * DV, device=dev) * 0.3).bfloat16()
    g = (-(torch.rand(L, H, device=dev) * 2 + 0.1)).float()
    beta = torch.rand(L, H, device=dev).bfloat16()
    cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
    seq_idx = torch.zeros(1, dtype=torch.int32, device=dev)
    state = torch.zeros(1, 1, H, DK, DV, dtype=torch.float32, device=dev)
    o = torch.empty(L, H * DV, dtype=qkv.dtype, device=dev)
    _gdn_recurrent_prefill_kernel[(1, H)](
        qkv, g, beta, state, o, cu, seq_idx,
        H=H, DK=DK, DV=DV, N_GDN=1, GDN_L=0, SCALE=SCALE, BLOCK_D=128)
    st_triton = state[0, 0].clone()

    # 纯 torch（正确 kv_mem = S.T @ k）
    q = qkv[:, :H * DK].float().reshape(L, H, DK)
    k = qkv[:, H * DK:2 * H * DK].float().reshape(L, H, DK)
    v = qkv[:, 2 * H * DK:].float().reshape(L, H, DV)
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = torch.zeros(H, DK, DV, device=dev)
    for i in range(L):
        qt = qn[i]; kt = kn[i]; vt = v[i]
        gt = g[i].exp(); bt = beta[i].float()
        S = S * gt.unsqueeze(-1).unsqueeze(-1)
        kv_mem = (S * kt.unsqueeze(-1)).sum(-2)  # [H,DV] 正确
        delta = (vt - kv_mem) * bt.unsqueeze(-1)
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
    d = (st_triton - S).abs()
    print("triton vs puretorch STATE: max_diff=%.5f cos=%.5f" % (
        d.max().item(), torch.nn.functional.cosine_similarity(
            st_triton.flatten().float(), S.flatten().float(), dim=0).item()))
    print("  triton_norm=%.4f torch_norm=%.4f" % (st_triton.float().norm().item(), S.float().norm().item()))
    for hh in range(H):
        print("  head %d: max_diff=%.5f" % (hh, (st_triton[hh] - S[hh]).abs().max().item()))


if __name__ == "__main__":
    main()
