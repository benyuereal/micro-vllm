import torch, triton, triton.language as tl

# 2-iteration loop, S_m loop-carried. Dump S_m and kv_mem at i=1.
@triton.jit
def loop_sum(S0, GE, K, DUMP, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.load(S0 + dk[:,None]*DV + dv[None,:]).to(tl.float32)  # S after token 0
    ge = tl.load(GE).to(tl.float32)
    S_m = S_m * ge
    k = tl.load(K + dk).to(tl.float32)
    tl.store(DUMP + 0*DK*DV + dk[:,None]*DV + dv[None,:], S_m)
    kv = tl.sum(S_m * k[None,:], axis=0)
    tl.store(DUMP + 1*DK*DV + dv, kv)

torch.manual_seed(0)
DK=DV=128
S0=torch.randn(DK,DV,device='cuda',dtype=torch.float32)*0.8
ge=torch.tensor([0.5],device='cuda',dtype=torch.float32)
K=torch.randn(DK,device='cuda',dtype=torch.float32)*0.088
DUMP=torch.zeros(2,DK*DV,device='cuda',dtype=torch.float32)
loop_sum[(1,)](S0,ge,K,DUMP,DK=DK,DV=DV)
Ssc=(S0*ge.item())
kv_ref=(Ssc*K[None,:]).sum(0)
print('Ssc: max_diff=%.3e' % (DUMP[0].reshape(DK,DV)-Ssc).abs().max())
print('kv:  max_diff=%.3e' % (DUMP[1].reshape(DV)-kv_ref).abs().max())
