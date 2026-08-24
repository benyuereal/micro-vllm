"""决定性测试：real kernel 的 state 到底匹配 正确kv_mem 还是 错误kv_mem(k[None,:]) 的纯torch。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from models.qwen3_5.adapter import _gdn_recurrent_prefill_kernel


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
    st_real = state[0, 0].clone()

    q = qkv[:, :H * DK].float().reshape(L, H, DK)
    k = qkv[:, H * DK:2 * H * DK].float().reshape(L, H, DK)
    v = qkv[:, 2 * H * DK:].float().reshape(L, H, DV)
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    def run(use_wrong):
        S = torch.zeros(H, DK, DV, device=dev)
        for i in range(L):
            kt = kn[i]; vt = v[i]
            gt = g[i].exp(); bt = beta[i].float()
            S = S * gt.unsqueeze(-1).unsqueeze(-1)
            if use_wrong:
                # k[None,:] 风格: kv_mem[j] = k[j] * sum_i S[i,j]
                kv_mem = (S * kt.unsqueeze(-1)).sum(-2)  # 正确
                kv_mem = kt * S.sum(-2)  # 错误 (k[j]*sum_i S[i,j])
            else:
                kv_mem = (S * kt.unsqueeze(-1)).sum(-2)  # 正确
            delta = (vt - kv_mem) * bt.unsqueeze(-1)
            S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        return S

    st_correct = run(False)
    st_wrong = run(True)
    dc = (st_real - st_correct).abs().max().item()
    dw = (st_real - st_wrong).abs().max().item()
    cc = torch.nn.functional.cosine_similarity(st_real.flatten(), st_correct.flatten(), dim=0).item()
    cw = torch.nn.functional.cosine_similarity(st_real.flatten(), st_wrong.flatten(), dim=0).item()
    print("real vs CORRECT-kv_mem: max_diff=%.5f cos=%.5f" % (dc, cc))
    print("real vs WRONG-kv_mem  : max_diff=%.5f cos=%.5f" % (dw, cw))


if __name__ == "__main__":
    main()
