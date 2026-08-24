"""验证 triton k[None,:] vs k[:,None] 广播方向。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch, triton
import triton.language as tl


@triton.jit
def test_kernel(S, k, OUT_A, OUT_B, DK: tl.constexpr, DV: tl.constexpr, BLOCK: tl.constexpr):
    dk = tl.arange(0, BLOCK)
    dv = tl.arange(0, BLOCK)
    S_m = tl.load(S + dk[:, None] * DV + dv[None, :]).to(tl.float32)
    k1 = tl.load(k + dk).to(tl.float32)
    a = tl.sum(S_m * k1[None, :], axis=0)   # 期望 [DV]
    b = tl.sum(S_m * k1[:, None], axis=0)   # 期望 [DV]
    tl.store(OUT_A + dv, a)
    tl.store(OUT_B + dv, b)


def main():
    DK = DV = 128
    torch.manual_seed(0)
    S = torch.randn(DK, DV, device="cuda")
    k = torch.randn(DK, device="cuda")
    out_a = torch.empty(DV, device="cuda")
    out_b = torch.empty(DV, device="cuda")
    test_kernel[(1,)](S, k, out_a, out_b, DK=DK, DV=DV, BLOCK=128)
    correct = (S.T @ k)  # kv_mem[j] = sum_i S[i,j]*k[i]
    print("A (k[None,:]) vs correct: max_diff=%.5f" % (out_a - correct).abs().max().item())
    print("B (k[:,None]) vs correct: max_diff=%.5f" % (out_b - correct).abs().max().item())


if __name__ == "__main__":
    main()
