import torch, triton, triton.language as tl

@triton.jit
def l2_test(X, OUT, DK: tl.constexpr, SCALE):
    dk = tl.arange(0, DK)
    q = tl.load(X + dk).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * SCALE
    tl.store(OUT + dk, q)

torch.manual_seed(0)
DK=128; scale=DK**-0.5
# bf16 input like conv output
x=torch.randn(DK,device='cuda',dtype=torch.bfloat16)
out=torch.empty(DK,device='cuda',dtype=torch.float32)
l2_test[(1,)](x,out,DK=DK,SCALE=scale)
xf=x.float()
ref=xf*torch.rsqrt((xf*xf).sum()+1e-6)*scale
print('l2norm Triton vs torch: max_diff=%.3e mean=%.3e' % ((out-ref).abs().max(),(out-ref).abs().mean()))
# also test rsqrt alone
@triton.jit
def rsqrt_test(X, OUT, DK: tl.constexpr):
    dk = tl.arange(0, DK)
    q = tl.load(X + dk).to(tl.float32)
    s = tl.sum(q*q)
    tl.store(OUT, s)
s_out=torch.empty(1,device='cuda',dtype=torch.float32)
rsqrt_test[(1,)](x,s_out,DK=DK)
print('sum(q*q) Triton=%.8f torch=%.8f diff=%.3e' % (s_out.item(), (xf*xf).sum().item(), abs(s_out.item()-(xf*xf).sum().item())))
