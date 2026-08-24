import torch, triton, triton.language as tl

# 2-iter real loop, H=16. For head 1 at i=1: dump S_m, k, and kv_mem.
@triton.jit
def loop_sum(QKV, G, BETA, DUMP, L, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
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
        if i == 1 and h == 1:
            tl.store(DUMP + 0*DK*DV + dk[:,None]*DV + dv[None,:], S_m)
            tl.store(DUMP + 1*DK*DV + dk, k)
        kv_mem = tl.sum(S_m * k[None,:], axis=0)
        if i == 1 and h == 1:
            tl.store(DUMP + 2*DK*DV + dv, kv_mem)
        delta = (v - kv_mem) * beta
        S_m += k[:,None] * delta[None,:]

torch.manual_seed(0)
L, H, DK, DV = 2, 16, 128, 128
scale = DK**-0.5
qkv = torch.randn(L, 2*H*DK + H*DV, device='cuda', dtype=torch.bfloat16)
g = -torch.rand(L, H, device='cuda', dtype=torch.float32)*2
beta = torch.rand(L, H, device='cuda', dtype=torch.bfloat16)
DUMP = torch.zeros(3, DK*DV, device='cuda', dtype=torch.float32)
loop_sum[(H,)](qkv, g, beta, DUMP, L, H=H, DK=DK, DV=DV, SCALE=scale, BLOCK_D=128)
Ssc = DUMP[0].reshape(DK,DV)
k1 = DUMP[1][:DK]
kv_kernel = DUMP[2][:DV]
kv_from_dump = (Ssc * k1[None,:]).sum(0)
print('kv_kernel vs torch-from-dumped(Ssc,k1): max_diff=%.3e' % (kv_kernel-kv_from_dump).abs().max())
print('  (if ~0, tl.sum is consistent with its own S_m,k; divergence is upstream)')
