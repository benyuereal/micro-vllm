"""Feed kernel + clean-torch the SAME pre-l2normed q/k/v (fp32). Isolate pure recurrence."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, triton
from models.qwen3_5.adapter import _gdn_recurrent_prefill_kernel

torch.manual_seed(0)
L, H, DK, DV = 5, 16, 128, 128
scale = DK**-0.5
# NON-unit q/k: kernel l2norms in fp32. ref = torch fp32 l2norm.
uq = torch.randn(L, H, DK, device='cuda', dtype=torch.float32)
uk = torch.randn(L, H, DK, device='cuda', dtype=torch.float32)
qn = uq/torch.sqrt((uq*uq).sum(-1,keepdim=True)+1e-6) * scale   # torch fp32 l2norm
kn = uk/torch.sqrt((uk*uk).sum(-1,keepdim=True)+1e-6)
v  = torch.randn(L, H, DV, device='cuda', dtype=torch.float32)
g  = -torch.rand(L, H, device='cuda', dtype=torch.float32)*2
beta = torch.rand(L, H, device='cuda', dtype=torch.float32)

# clean torch (no l2norm, qn already scaled)
S = torch.zeros(H, DK, DV, device='cuda')
outs=[]
for i in range(L):
    S = S*g[i].exp().view(H,1,1)
    kv_mem = (S*kn[i].unsqueeze(-1)).sum(-2)
    delta = (v[i]-kv_mem)*beta[i].view(H,1)
    S = S + kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
    outs.append((S*qn[i].unsqueeze(-1)).sum(-2))
ref = torch.stack(outs)  # [L,H,DV]

# kernel: build qkv in bf16 (kernel casts to fp32). q/k already l2normed+scaled.
# kernel layout per token: [q(H*DK) | k(H*DK) | v(H*DV)]
qkv = torch.cat([qn.reshape(L,-1), kn.reshape(L,-1), v.reshape(L,-1)], dim=1).to(torch.bfloat16).contiguous()
g_m = g.contiguous(); beta_m = beta.to(torch.bfloat16).contiguous()
cu_q = torch.tensor([0,L],dtype=torch.int32,device='cuda')
seq_idx = torch.zeros(1,dtype=torch.int32,device='cuda')
state = torch.zeros(1,18,H,DK,DV,dtype=torch.float32,device='cuda')
o = torch.empty(L,H*DV,dtype=torch.bfloat16,device='cuda')
_gdn_recurrent_prefill_kernel[(1,H)](qkv,g_m,beta_m,state,o,cu_q,seq_idx,
    H=H,DK=DK,DV=DV,N_GDN=18,GDN_L=0,SCALE=scale,BLOCK_D=128)
o = o.reshape(L,H,DV)
d=(ref-o.float()).abs()
print('kernel vs clean-torch (same pre-normed input): max_diff=%.6f cos=%.6f' % (d.max(), F.cosine_similarity(ref.flatten(),o.flatten().float(),dim=0)))
for i in range(L):
    print('  t=%d max_diff=%.6f' % (i,(ref[i]-o[i].float()).abs().max()))
