"""Verify: bf16-l2norm + fp32-recurrence (clean torch) matches HF torch_recurrent?"""
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, torch_recurrent_gated_delta_rule
MODEL='/models/Qwen3.5-0.8B'
tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
hf=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',trust_remote_code=True,local_files_only=True).eval()
ids=tok.encode('The capital of France is',add_special_tokens=True)
L=len(ids); text=hf.model; la=text.layers[0].linear_attn
with torch.no_grad():
    h=text.embed_tokens(torch.tensor([ids],device='cuda:0'))[0]
    normed=text.layers[0].input_layernorm(h.unsqueeze(0))[0]
    mqkv=la.in_proj_qkv(normed.unsqueeze(0))
    b=la.in_proj_b(normed.unsqueeze(0)); a=la.in_proj_a(normed.unsqueeze(0))
    conv=causal_conv1d_fn(mqkv.transpose(1,2),la.conv1d.weight.squeeze(1),la.conv1d.bias,activation=la.activation).transpose(1,2)
    beta=b.sigmoid(); g=-la.A_log.float().exp()*F.softplus(a.float()+la.dt_bias)
    kd,vd=la.key_dim,la.value_dim
    q,k,v=torch.split(conv,[kd,kd,vd],dim=-1)
    q=q.reshape(L,-1,la.head_k_dim);k=k.reshape(L,-1,la.head_k_dim);v=v.reshape(L,-1,la.head_v_dim)
    H,DK,DV=16,128,128; scale=DK**-0.5
    # HF ground truth
    core_hf,_=torch_recurrent_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g=g,beta=beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    core_hf=core_hf[0]
    # Variant A: bf16 l2norm (match HF) + fp32 recurrence
    def l2b(x): return x*torch.rsqrt((x*x).sum(-1,keepdim=True)+1e-6)  # bf16
    qn=l2b(q).float()*scale; kn=l2b(k).float()
    vf=v.float(); betaf=beta[0].float(); gf=g[0].float()
    S=torch.zeros(H,DK,DV,device='cuda'); outs=[]
    for i in range(L):
        S=S*gf[i].exp().view(H,1,1)
        kv_mem=(S*kn[i].unsqueeze(-1)).sum(-2)
        delta=(vf[i]-kv_mem)*betaf[i].view(H,1)
        S=S+kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
        outs.append((S*qn[i].unsqueeze(-1)).sum(-2))
    coreA=torch.stack(outs)
    # Variant B: fp32 l2norm + fp32 recurrence (my kernel)
    qf=q.float(); kf=k.float()
    qn2=qf*torch.rsqrt((qf*qf).sum(-1,keepdim=True)+1e-6)*scale
    kn2=kf*torch.rsqrt((kf*kf).sum(-1,keepdim=True)+1e-6)
    S2=torch.zeros(H,DK,DV,device='cuda'); outs2=[]
    for i in range(L):
        S2=S2*gf[i].exp().view(H,1,1)
        kv_mem=(S2*kn2[i].unsqueeze(-1)).sum(-2)
        delta=(vf[i]-kv_mem)*betaf[i].view(H,1)
        S2=S2+kn2[i].unsqueeze(-1)*delta.unsqueeze(-2)
        outs2.append((S2*qn2[i].unsqueeze(-1)).sum(-2))
    coreB=torch.stack(outs2)
    print('HF core norm=%.5f' % core_hf.float().norm().item())
    print('A (bf16 l2norm): max_diff=%.6f cos=%.6f' % ((core_hf.float()-coreA).abs().max(), F.cosine_similarity(core_hf.flatten().float(),coreA.flatten(),dim=0)))
    print('B (fp32 l2norm): max_diff=%.6f cos=%.6f' % ((core_hf.float()-coreB).abs().max(), F.cosine_similarity(core_hf.flatten().float(),coreB.flatten(),dim=0)))
