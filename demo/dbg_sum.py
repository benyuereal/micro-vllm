import torch, triton, triton.language as tl

@triton.jit
def sum_test(S, K, OUT, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.load(S + dk[:,None]*DV + dv[None,:]).to(tl.float32)
    k = tl.load(K + dk).to(tl.float32)
    kv = tl.sum(S_m * k[None,:], axis=0)
    tl.store(OUT + dv, kv)

@triton.jit
def sum_test2(S, Q, OUT, DK: tl.constexpr, DV: tl.constexpr):
    dk = tl.arange(0, DK); dv = tl.arange(0, DV)
    S_m = tl.load(S + dk[:,None]*DV + dv[None,:]).to(tl.float32)
    q = tl.load(Q + dk).to(tl.float32)
    o = tl.sum(S_m * q[:,None], axis=0)
    tl.store(OUT + dv, o)

torch.manual_seed(0)
DK=DV=128
S=torch.randn(DK,DV,device='cuda',dtype=torch.float32)*0.01
K=torch.randn(DK,device='cuda',dtype=torch.float32)*0.088
out=torch.empty(DV,device='cuda',dtype=torch.float32)
sum_test[(1,)](S,K,out,DK=DK,DV=DV)
ref=(S*K[None,:]).sum(0)
print('tl.sum kv_mem [128,128]: max_diff=%.3e mean=%.3e' % ((out-ref).abs().max(), (out-ref).abs().mean()))
Q=torch.randn(DK,device='cuda',dtype=torch.float32)*0.088
out2=torch.empty(DV,device='cuda',dtype=torch.float32)
sum_test2[(1,)](S,Q,out2,DK=DK,DV=DV)
ref2=(S*Q[:,None]).sum(0)
print('tl.sum o [128,128]: max_diff=%.3e mean=%.3e' % ((out2-ref2).abs().max(), (out2-ref2).abs().mean()))
