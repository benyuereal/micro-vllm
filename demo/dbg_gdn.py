"""GDN layer-0 中间量对比：micro vs HF（eager 复现 HF GDN forward）。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/models/Qwen3.5-0.8B"
PROMPT = "The capital of France is"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                          device_map="cuda:0", trust_remote_code=True,
                                          local_files_only=True)
hf.eval()
ids = tok.encode(PROMPT, add_special_tokens=True)
x = torch.tensor([ids], device="cuda:0")

# ---- HF layer 0 GDN intermediates (eager) ----
text = hf.model  # Qwen3_5TextModel
emb = text.embed_tokens(x)  # [1, T, hidden]
la = text.layers[0].linear_attn
in_ln = text.layers[0].input_layernorm
# 1-centered RMSNorm
def rms1(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)
h0 = rms1(emb, in_ln.weight, in_ln.eps)  # [1,T,hidden]

mixed_qkv_pre = la.in_proj_qkv(h0)          # [1,T,6144]
z = la.in_proj_z(h0)                        # [1,T,2048]
b = la.in_proj_b(h0)                        # [1,T,16]
a = la.in_proj_a(h0)                        # [1,T,16]
beta = b.sigmoid()
g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)

# conv (causal, kernel 4, groups, silu) — HF causal_conv1d_fn
def causal_conv1d_fn(hs, weight, activation):
    _, hidden_size, seq_len = hs.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hs.to(weight.dtype), weight=weight.unsqueeze(1), bias=None,
                   padding=padding, groups=hidden_size)[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hs.dtype)
mixed_qkv_post = causal_conv1d_fn(mixed_qkv_pre.transpose(1,2), la.conv1d.weight.squeeze(1), "silu").transpose(1,2)

# split q/k/v
key_dim = la.key_dim; value_dim = la.value_dim
query, key, value = torch.split(mixed_qkv_post, [key_dim, key_dim, value_dim], dim=-1)
query = query.reshape(1, -1, la.num_k_heads, la.head_k_dim)
key = key.reshape(1, -1, la.num_k_heads, la.head_k_dim)
value = value.reshape(1, -1, la.num_v_heads, la.head_v_dim)

# recurrent: use HF's ACTUAL torch_recurrent_gated_delta_rule (true reference)
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
o_hf, _ = torch_recurrent_gated_delta_rule(
    query, key, value, g=g, beta=beta,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True)
o_hf = o_hf.reshape(1, -1, la.num_v_heads * la.head_v_dim)  # [1, T, H*DV]

# norm_gated: out = (o * rrms * w) * silu(z), per (token, head) on DV
def norm_gated(o, z, w, eps):
    # o, z: [1, T, H*DV]; w: [DV]
    H, DV = la.num_v_heads, la.head_v_dim
    of = o.float().reshape(1, -1, H, DV)
    zf = z.float().reshape(1, -1, H, DV)
    var = of.pow(2).mean(-1, keepdim=True)
    of = of * torch.rsqrt(var + eps)
    of = (w.float() * of).to(o.dtype)
    of = of * F.silu(zf.to(o.dtype).float()).to(o.dtype)
    return of.reshape(1, -1, H * DV).to(o.dtype)
og_hf = norm_gated(o_hf.to(torch.bfloat16), z, la.norm.weight, la.norm.variance_epsilon)
out_hf = la.out_proj(og_hf)  # [1, T, hidden]

# ---- micro ----
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
pad = eng.prefill_runner.adapter
pad._dbg_gdn = []
# 诊断：warmup 后状态池是否有残留
sp = eng.prefill_runner._gdn_state_pool
print(f"[diag] after warmup: state_pool max={sp.abs().max().item():.4f} "
      f"nonzero_slots={ (sp.abs().sum(dim=(1,2,3,4)) > 0).sum().item() } / {sp.shape[0]}")
eng.add_request(PROMPT, 1, temperature=0.0, top_p=1.0)
b2, bt2 = eng.get_next_batch()
seq0 = b2[0]
print(f"[diag] real seq_id={seq0.seq_id} prefill_done={seq0.prefill_done} "
      f"_gdn_slot={getattr(seq0,'_gdn_slot',None)}")
ctx = BatchInferenceContext(len(b2), bt2, b2)
eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
# 诊断：kernel 实际读的 seq_idx buffer + 分配的 slot
slot = getattr(seq0, "_gdn_slot", None)
print(f"[diag] after step: _gdn_prefill_seq_idx={eng.prefill_runner._gdn_prefill_seq_idx[:1].tolist()} "
      f"seq0._gdn_slot={slot}")
nz = (sp.abs().sum(dim=(1,2,3,4)) > 0).nonzero().flatten().tolist()
print(f"[diag] after prefill: nonzero slots={nz} (real slot should be {slot})")
mc = pad._dbg_gdn[0]  # layer 0

def cmp(name, hf_t, mc_t):
    hf_t = hf_t.float().reshape(-1)
    mc_t = mc_t.float().reshape(-1)
    n = min(hf_t.numel(), mc_t.numel())
    d = (hf_t[:n] - mc_t[:n]).abs().max().item()
    rel = d / (hf_t[:n].abs().max().item() + 1e-6)
    print(f"  {name:14s} maxdiff={d:.5f} rel={rel:.5f}  hf_max={hf_t[:n].abs().max().item():.3f}")

print("=== GDN layer 0 intermediates (last token) ===")
# micro qkv_pre is [M, 6144] where M=T (single seq). last token = index T-1
Tm = mc["qkv_pre"].shape[0]
cmp("qkv_pre", mixed_qkv_pre[0, -1], mc["qkv_pre"][Tm-1])
cmp("z", z[0, -1], mc["z"][Tm-1])
cmp("b", b[0, -1], mc["b"][Tm-1])
cmp("a", a[0, -1], mc["a"][Tm-1])
cmp("g", g[0, -1], mc["g"][Tm-1])
cmp("beta", beta[0, -1], mc["beta"][Tm-1])
cmp("qkv_post", mixed_qkv_post[0, -1], mc["qkv_post"][Tm-1])
cmp("o", o_hf[0, -1], mc["o"][Tm-1])
# 隔离 recurrent：用 micro 的 qkv_post（bf16）喂 HF 的 recurrent，对比 micro 的 o
mc_qkv = mc["qkv_post"]  # [T, 6144] bf16
mq = mc_qkv[:, :key_dim].reshape(Tm, la.num_k_heads, la.head_k_dim).unsqueeze(0)
mk = mc_qkv[:, key_dim:2*key_dim].reshape(Tm, la.num_k_heads, la.head_k_dim).unsqueeze(0)
mv = mc_qkv[:, 2*key_dim:].reshape(Tm, la.num_v_heads, la.head_v_dim).unsqueeze(0)
mg = mc["g"].unsqueeze(0)  # [1, T, H]
mb = mc["beta"].unsqueeze(0)
o_from_mc, _ = torch_recurrent_gated_delta_rule(
    mq, mk, mv, g=mg, beta=mb, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True)
o_from_mc = o_from_mc.reshape(1, Tm, -1)
cmp("o(from mc qkv)", o_from_mc[0, -1], mc["o"][Tm-1])

# 纯 fp32 递推，l2norm 分别用 fp32 / bf16，喂 micro 的 qkv_post，对比 micro 的 o
def rec_fp32(qkv, l2norm_dtype):
    q = qkv[:, :key_dim].reshape(Tm, la.num_k_heads, la.head_k_dim).float()
    k = qkv[:, key_dim:2*key_dim].reshape(Tm, la.num_k_heads, la.head_k_dim).float()
    v = qkv[:, 2*key_dim:].reshape(Tm, la.num_v_heads, la.head_v_dim).float()
    gg = mc["g"].float(); bb = mc["beta"].float()
    def l2n(x, dt):
        xd = x.to(dt)
        inv = torch.rsqrt((xd*xd).sum(-1, keepdim=True) + 1e-6)
        return (xd*inv).float()
    q = l2n(q, l2norm_dtype) * (la.head_k_dim ** -0.5)
    k = l2n(k, l2norm_dtype)
    S = torch.zeros(la.num_v_heads, la.head_k_dim, la.head_v_dim, device="cuda:0")
    outs = []
    for t in range(Tm):
        qt=q[t]; kt=k[t]; vt=v[t]; ge=gg[t].exp().view(-1,1,1); bt=bb[t].view(-1,1)
        S = S * ge
        kv_mem = (S * kt.unsqueeze(-1)).sum(-2)
        delta = (vt - kv_mem) * bt
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((S * qt.unsqueeze(-1)).sum(-2))
    return torch.stack(outs, dim=0)  # [T, H*DV]
o_fp32 = rec_fp32(mc_qkv, torch.float32)
o_bf16 = rec_fp32(mc_qkv, torch.bfloat16)
cmp("o(fp32 l2n)", o_fp32[Tm-1], mc["o"][Tm-1])
cmp("o(bf16 l2n)", o_bf16[Tm-1], mc["o"][Tm-1])
cmp("og", og_hf[0, -1], mc["og"][Tm-1])
cmp("out", out_hf[0, -1], mc["out"][Tm-1])
