"""3-way: kernel vs clean-torch(fp32 l2norm) vs HF, on real conv_out data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F, triton
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, torch_recurrent_gated_delta_rule
from models.qwen3_5.adapter import _gdn_recurrent_prefill_kernel
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
    core_hf,_=torch_recurrent_gated_delta_rule(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g=g,beta=beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    core_hf=core_hf[0]
    # clean-torch fp32 l2norm
    qf=q.float(); kf=k.float()
    qn=qf*torch.rsqrt((qf*qf).sum(-1,keepdim=True)+1e-6)*scale
    kn=kf*torch.rsqrt((kf*kf).sum(-1,keepdim=True)+1e-6)
    vf=v.float(); betaf=beta[0].float(); gf=g[0].float()
    S=torch.zeros(H,DK,DV,device='cuda'); outs=[]
    for i in range(L):
        S=S*gf[i].exp().view(H,1,1)
        kv_mem=(S*kn[i].unsqueeze(-1)).sum(-2)
        delta=(vf[i]-kv_mem)*betaf[i].view(H,1)
        S=S+kn[i].unsqueeze(-1)*delta.unsqueeze(-2)
        outs.append((S*qn[i].unsqueeze(-1)).sum(-2))
    core_torch=torch.stack(outs)
    # kernel
    qkv=conv[0].contiguous()
    g_m=g[0].contiguous(); beta_m=beta[0].contiguous()
    cu_q=torch.tensor([0,L],dtype=torch.int32,device='cuda')
    seq_idx=torch.zeros(1,dtype=torch.int32,device='cuda')
    state=torch.zeros(1,18,H,DK,DV,dtype=torch.float32,device='cuda')
    o=torch.empty(L,H*DV,dtype=qkv.dtype,device='cuda')
    _gdn_recurrent_prefill_kernel[(1,H)](qkv,g_m,beta_m,state,o,cu_q,seq_idx,
        H=H,DK=DK,DV=DV,N_GDN=18,GDN_L=0,SCALE=scale,BLOCK_D=128)
    o=o.reshape(L,H,DV)
    def cmp(name,a,b):
        d=(a.float()-b.float()).abs().max()
        c=F.cosine_similarity(a.flatten().float(),b.flatten().float(),dim=0)
        print('%-22s max_diff=%.6f cos=%.6f' % (name,d,c))
    cmp('kernel vs HF', o, core_hf)
    cmp('torch  vs HF', core_torch, core_hf)
    cmp('kernel vs torch', o, core_torch)
    # per-token kernel vs torch
    for i in range(L):
        print('  t=%d kernel-vs-torch=%.6f kernel-vs-HF=%.6f' % (
            i,(o[i].float()-core_torch[i]).abs().max(),(o[i].float()-core_hf[i].float()).abs().max()))
