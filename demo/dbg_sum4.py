import torch, triton, triton.language as tl

# S_m is loop-carried (built via += in iter 0), then *ge, then tl.sum at i=1.
@triton.jit
def loop_sum(S0, GE, K, DUMP, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.zeros([DK, DV], dtype=tl.float32)
    # iter 0: build S_m via += (like the real kernel)
    k0 = tl.load(K + dk).to(tl.float32)
    d0 = tl.load(K + dv).to(tl.float32)  # dummy delta
    S_m += k0[:,None] * d0[None,:]
    # iter 1
    ge = tl.load(GE).to(tl.float32)
    S_m = S_m * ge
    k1 = tl.load(K + DK + dk).to(tl.float32)
    tl.store(DUMP + 0*DK*DV + dk[:,None]*DV + dv[None,:], S_m)
    kv = tl.sum(S_m * k1[None,:], axis=0)
    tl.store(DUMP + 1*DK*DV + dv, kv)

torch.manual_seed(0)
DK=DV=128
K=torch.randn(2*DK,device='cuda',dtype=torch.float32)*0.088
ge=torch.tensor([0.5],device='cuda',dtype=torch.float32)
DUMP=torch.zeros(2,DK*DV,device='cuda',dtype=torch.float32)
loop_sum[(1,)](None,ge,K,DUMP,DK=DK,DV=DV)
# torch: S0 = outer(k0, d0); Ssc = S0*ge; kv = Ssc @ k1
k0=K[:DK]; d0=K[:DK]; k1=K[DK:2*DK]
S0=torch.outer(k0,d0)
Ssc=S0*ge.item()
kv_ref=(Ssc*k1[None,:]).sum(0)
print('Ssc: max_diff=%.3e' % (DUMP[0].reshape(DK,DV)-Ssc).abs().max())
print('kv:  max_diff=%.3e' % (DUMP[1][:DV]-kv_ref).abs().max())
