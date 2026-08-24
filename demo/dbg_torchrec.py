"""Clean standalone torch GDN recurrence; verify against HF torch_recurrent_gated_delta_rule."""
import torch, torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

torch.manual_seed(0)
L, H, DK, DV = 5, 16, 128, 128
scale = DK**-0.5
q = torch.randn(L, H, DK, device='cuda', dtype=torch.bfloat16)
k = torch.randn(L, H, DK, device='cuda', dtype=torch.bfloat16)
v = torch.randn(L, H, DV, device='cuda', dtype=torch.bfloat16)
g = -torch.rand(L, H, device='cuda', dtype=torch.float32)*2
beta = torch.rand(L, H, device='cuda', dtype=torch.bfloat16)

# HF ground truth (bf16 l2norm -> fp32)
core_hf, state_hf = torch_recurrent_gated_delta_rule(
    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0),
    initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
core_hf = core_hf[0]  # [L,H,DV]

# My clean torch (fp32 l2norm)
def l2b(x): return x*torch.rsqrt((x*x).sum(-1,keepdim=True)+1e-6)
qn = l2b(q).float()*scale
kn = l2b(k).float()
vf = v.float(); betaf = beta.float(); gf = g.float()
S = torch.zeros(H, DK, DV, device='cuda')
outs = []
for i in range(L):
    S = S*gf[i].exp().view(H,1,1)
    kv_mem = (S*kn[i].unsqueeze(-1)).sum(-2)
    delta = (vf[i]-kv_mem)*betaf[i].view(H,1)
    S = S + kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
    outs.append((S*qn[i].unsqueeze(-1)).sum(-2))
core_mine = torch.stack(outs)
d = (core_hf.float()-core_mine).abs()
print('clean-torch vs HF: max_diff=%.6f cos=%.6f' % (d.max(), F.cosine_similarity(core_hf.flatten().float(),core_mine.flatten(),dim=0)))
for i in range(L):
    print('  t=%d max_diff=%.6f' % (i,(core_hf[i].float()-core_mine[i]).abs().max()))
# self-check: is kv_mem consistent?
S2 = torch.zeros(H, DK, DV, device='cuda')
for i in range(L):
    S2 = S2*gf[i].exp().view(H,1,1)
    if i==1: Ssc=S2
    kv_mem = (S2*kn[i].unsqueeze(-1)).sum(-2)
    if i==1:
        selfchk = (Ssc[1]*kn[1,1][None,:]).sum(0)
        print('SELF-CHECK kv_mem[1,1] vs (Ssc*kn).sum: max_diff=%.3e' % (kv_mem[1]-selfchk).abs().max())
    delta = (vf[i]-kv_mem)*betaf[i].view(H,1)
    S2 = S2 + kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
