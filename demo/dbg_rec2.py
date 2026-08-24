"""Dump each sub-step at t=1 for head 0: S0, S0*exp(g1), kv_mem, delta, S1."""
import torch, triton, triton.language as tl

@triton.jit
def rec_sub(QKV, G, BETA, DUMP, L, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
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
        if i == 1:
            tl.store(DUMP + 0*DK*DV + dk[:,None]*DV + dv[None,:], S_m)  # S0*exp(g1)
        kv_mem = tl.sum(S_m * k[None,:], axis=0)
        if i == 1:
            tl.store(DUMP + 1*DK*DV + dv, kv_mem)
        delta = (v - kv_mem) * beta
        if i == 1:
            tl.store(DUMP + 2*DK*DV + dv, delta)
        S_m += k[:,None] * delta[None,:]
        if i == 1:
            tl.store(DUMP + 3*DK*DV + dk[:,None]*DV + dv[None,:], S_m)  # S1

torch.manual_seed(0)
L, H, DK, DV = 2, 1, 128, 128
scale = DK**-0.5
qkv = torch.randn(L, 2*H*DK + H*DV, device='cuda', dtype=torch.bfloat16)
g = -torch.rand(L, H, device='cuda', dtype=torch.float32)*2
beta = torch.rand(L, H, device='cuda', dtype=torch.bfloat16)
DUMP = torch.zeros(4, DK, DV, device='cuda', dtype=torch.float32)
rec_sub[(H,)](qkv, g, beta, DUMP, L, H=H, DK=DK, DV=DV, SCALE=scale, BLOCK_D=128)

# torch
q = qkv[:, :H*DK].reshape(L, H, DK).float()
k = qkv[:, H*DK:2*H*DK].reshape(L, H, DK).float()
v = qkv[:, 2*H*DK:].reshape(L, H, DV).float()
qn = q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)*scale
kn = k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
S = torch.zeros(H, DK, DV, device='cuda')
for i in range(L):
    S = S*g[i].exp().view(H,1,1)
    if i==1:
        S_scaled = S
    kv_mem = (S*kn[i].unsqueeze(-1)).sum(-2)
    delta = (v[i]-kv_mem)*beta[i].float().view(H,1)
    S = S + kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
    if i==1:
        S1 = S
# compare
print('S0*exp(g1): max_diff=%.6f' % (DUMP[0]-S_scaled[0]).abs().max())
print('kv_mem:     max_diff=%.6f' % (DUMP[1]-kv_mem[0]).abs().max())
print('delta:      max_diff=%.6f' % (DUMP[2]-delta[0]).abs().max())
print('S1:         max_diff=%.6f' % (DUMP[3]-S1[0]).abs().max())
print('S1 norm kernel=%.4f torch=%.4f' % (DUMP[3].norm(), S1[0].norm()))
