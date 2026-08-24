"""Clean recurrence kernel that dumps S after each token; compare to torch step-by-step."""
import torch, triton, triton.language as tl

@triton.jit
def rec_dump(QKV, G, BETA, S_DUMP, L, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             SCALE, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    dk = tl.arange(0, BLOCK_D); dv = tl.arange(0, BLOCK_D)
    S_m = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    for i in range(0, L):
        q_base = QKV + i.to(tl.int64) * (2*H*DK + H*DV) + h*DK
        k_base = q_base + H*DK
        v_base = k_base + H*DK
        q = tl.load(q_base + dk).to(tl.float32)
        k = tl.load(k_base + dk).to(tl.float32)
        v = tl.load(v_base + dv).to(tl.float32)
        g = tl.load(G + i*H + h).to(tl.float32)
        beta = tl.load(BETA + i*H + h).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q*q) + 1e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k*k) + 1e-6)
        ge = tl.exp(g)
        S_m = S_m * ge
        kv_mem = tl.sum(S_m * k[None,:], axis=0)
        delta = (v - kv_mem) * beta
        S_m += k[:,None] * delta[None,:]
        # dump S after this token
        tl.store(S_DUMP + (i*H + h).to(tl.int64)*DK*DV + dk[:,None]*DV + dv[None,:], S_m)

torch.manual_seed(0)
L, H, DK, DV = 5, 16, 128, 128
scale = DK**-0.5
# random bf16 qkv
qkv = torch.randn(L, 2*H*DK + H*DV, device='cuda', dtype=torch.bfloat16)
g = -torch.rand(L, H, device='cuda', dtype=torch.float32)*2
beta = torch.rand(L, H, device='cuda', dtype=torch.bfloat16)
S_dump = torch.zeros(L, H, DK, DV, device='cuda', dtype=torch.float32)
rec_dump[(H,)](qkv, g, beta, S_dump, L, H=H, DK=DK, DV=DV, SCALE=scale, BLOCK_D=128)

# torch replication
q = qkv[:, :H*DK].reshape(L, H, DK).float()
k = qkv[:, H*DK:2*H*DK].reshape(L, H, DK).float()
v = qkv[:, 2*H*DK:].reshape(L, H, DV).float()
qn = q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*scale
kn = k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
S = torch.zeros(H, DK, DV, device='cuda')
S_torch = torch.zeros(L, H, DK, DV, device='cuda')
for i in range(L):
    S = S*g[i].exp().view(H,1,1)
    kv_mem = (S*kn[i].unsqueeze(-1)).sum(-2)
    delta = (v[i]-kv_mem)*beta[i].float().view(H,1)
    S = S + kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
    S_torch[i] = S
d = (S_dump - S_torch).abs()
print('per-token state: max_diff=%.6f' % d.max())
for i in range(L):
    print('  t=%d max_diff=%.6f' % (i, (S_dump[i]-S_torch[i]).abs().max()))
# per-head at t=1
print('per-head at t=1:')
for h in range(H):
    dd = (S_dump[1,h]-S_torch[1,h]).abs().max()
    if dd > 1e-4:
        print('  head %d: max_diff=%.6f' % (h, dd))
