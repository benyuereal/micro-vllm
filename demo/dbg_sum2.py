import torch, triton, triton.language as tl

@triton.jit
def sum_loaded(S, K, OUT, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.load(S + dk[:,None]*DV + dv[None,:]).to(tl.float32)
    k = tl.load(K + dk).to(tl.float32)
    tl.store(OUT + dv, tl.sum(S_m * k[None,:], axis=0))

@triton.jit
def sum_computed(S, GE, K, OUT, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.load(S + dk[:,None]*DV + dv[None,:]).to(tl.float32)
    ge = tl.load(GE).to(tl.float32)
    S_m = S_m * ge
    k = tl.load(K + dk).to(tl.float32)
    tl.store(OUT + dv, tl.sum(S_m * k[None,:], axis=0))

torch.manual_seed(0)
DK=DV=128
S=torch.randn(DK,DV,device='cuda',dtype=torch.float32)*0.8
K=torch.randn(DK,device='cuda',dtype=torch.float32)*0.088
ge=torch.tensor([0.5],device='cuda',dtype=torch.float32)
o1=torch.empty(DV,device='cuda',dtype=torch.float32)
o2=torch.empty(DV,device='cuda',dtype=torch.float32)
sum_loaded[(1,)](S,K,o1,DK=DK,DV=DV)
sum_computed[(1,)](S,ge,K,o2,DK=DK,DV=DV)
ref_loaded=(S*K[None,:]).sum(0)
ref_comp=((S*ge.item())*K[None,:]).sum(0)
print('loaded:   max_diff=%.3e' % (o1-ref_loaded).abs().max())
print('computed: max_diff=%.3e' % (o2-ref_comp).abs().max())
print('loaded vs computed (should be ge*ref_loaded): max_diff=%.3e' % (o2-ge.item()*ref_loaded).abs().max())
