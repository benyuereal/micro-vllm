"""isolation 测 attention 段各 kernel 绝对时间（单层, 仅供参考量级, graph 下不可信但能看大小排序）。"""
import sys, torch
sys.path.insert(0, "/models/micro-vllm")
from kernel.pre_mla import get_pre_qkv_kernel, get_pre_kva_kernel, get_absorb_kernel
from kernel.rmsnorm import rmsnorm_
import torch.nn.functional as F

bs=1; H=2048; num_heads=16; q_head=576; qk_nope=512; kv_lora=512; qk_rope=64; v_head=128
latent_dim=576; block_size=16; max_seq_blocks=64; n_blocks=512
dtype=torch.bfloat16; dev="cuda"

torch.manual_seed(0)
x16 = torch.zeros(bs,16,H,dtype=dtype,device=dev); x16[:,0,:]=torch.randn(bs,H,dtype=dtype,device=dev)*0.1
q_w=torch.randn(num_heads*q_head,H,dtype=dtype,device=dev)*0.02; q_b=torch.zeros(num_heads*q_head,dtype=dtype,device=dev)
kva_w=torch.randn(kv_lora+qk_rope,H,dtype=dtype,device=dev)*0.02; kva_b=torch.zeros(kv_lora+qk_rope,dtype=dtype,device=dev)
cos_q=torch.randn(bs,qk_rope,dtype=dtype,device=dev); sin_q=torch.randn(bs,qk_rope,dtype=dtype,device=dev)
q_nope16=torch.randn(bs*num_heads,16,qk_nope,dtype=dtype,device=dev)*0.1
kvb_kn_t=torch.randn(num_heads,kv_lora,qk_nope,dtype=dtype,device=dev)*0.02
absorb_idx=torch.arange(bs*num_heads,dtype=torch.int32,device=dev)
o_w=torch.randn(H,num_heads*v_head,dtype=dtype,device=dev)*0.02; o_b=torch.zeros(H,dtype=dtype,device=dev)
attn_out=torch.randn(bs,num_heads*v_head,dtype=dtype,device=dev)*0.1
in_ln_w=torch.ones(H,dtype=dtype,device=dev)

kq=get_pre_qkv_kernel(bs,H,num_heads,q_head,qk_rope,dtype)
kk=get_pre_kva_kernel(bs,H,latent_dim,block_size,max_seq_blocks,n_blocks,dtype)
ka=get_absorb_kernel(bs,num_heads,qk_nope,kv_lora,dtype)

bt=torch.zeros(bs,max_seq_blocks,dtype=torch.int32,device=dev)
new_pos=torch.zeros(bs,dtype=torch.int32,device=dev)
k_cache=torch.zeros(n_blocks,block_size,1,latent_dim,dtype=dtype,device=dev)
v_cache=torch.zeros(n_blocks,block_size,1,latent_dim,dtype=dtype,device=dev)

def t(fn,iters=500):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000

print(f"rmsnorm:        {t(lambda: rmsnorm_(x16[:,0,:], in_ln_w, x16[:,0,:], 1e-6)):.1f} us")
print(f"pre_qkv:        {t(lambda: kq(x16,q_w,q_b,cos_q,sin_q)):.1f} us  (grid=144)")
q_out=kq(x16,q_w,q_b,cos_q,sin_q)
print(f"pre_kva:        {t(lambda: kk(x16,kva_w,kva_b,bt,new_pos,k_cache,v_cache)):.1f} us  (grid=9)")
qn16=q_out[:,:,:,:qk_nope].reshape(bs*num_heads,16,qk_nope).contiguous()
print(f"absorb:         {t(lambda: ka(qn16,kvb_kn_t,absorb_idx)):.1f} us  (grid=128)")
print(f"o_proj(F.linear):{t(lambda: F.linear(attn_out,o_w,o_b)):.1f} us  (grid~32)")
print(f"sum(非MLA):     rmsnorm+pre_qkv+pre_kva+absorb+o_proj")
