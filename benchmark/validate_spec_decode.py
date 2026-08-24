"""DFlash2 投机解码机制正确性验证（自起草：草稿=目标模型 Qwen3-0.6B）。

验证目标：
1. 投机解码（draft-verify-accept，greedy 确定性接受）输出与无投机解码 greedy 逐 token 一致。
2. 统计平均接受 token 数（acceptance length）。
3. 对比单用户 decode 吞吐（有/无投机解码）。

机制（对齐 vLLM dflash2，clean 索引）：
状态：有效 token 序列 tokens（长度 len(tokens)），KV cache 含 tokens[0:kv_len] 的 KV
（kv_len <= len(tokens)；最后一个 bonus token 的 KV 尚未写入，由下一步 anchor 补写）。

每步（N=投机 token 数）：
  anchor = tokens[kv_len-1]（KV 已在 cache 的最后一个 token）
  1. Draft：喂 [anchor, mask*N]（1+N token，非因果）给草稿模型，
     context = KV[0:kv_len-1]（anchor 之前）。N 个 mask 位置产出 d_0..d_{N-1}。
     自起草时草稿=目标，context KV 直接读目标 cache（草稿无需独立持久 KV）。
  2. Verify：喂 [anchor, d_0..d_{N-1}]（1+N token，因果）给目标模型，
     context = KV[0:kv_len-1]。写入 [anchor, d_0..d_{N-1}] 的 KV 到 [kv_len-1, kv_len-1+1+N)。
     得到 N+1 个 target 预测：target_preds[i] = p_{kv_len+i}（i=0..N）。
  3. Accept（greedy）：从 i=0 起，d_i==target_preds[i] 则接受，遇首个不一致停止；
     accepted=接受数，bonus=target_preds[accepted]。
     new_tokens = d_0..d_{accepted-1} + [bonus]（accepted+1 个）。
  4. tokens.extend(new_tokens)；kv_len += accepted（bonus 的 KV 下一步补写）。

正确性：接受的 token 全是 target 自己的预测（d_i==p_{kv_len+i} 或 bonus=p_{kv_len+accepted}），
故输出与 plain greedy 严格一致。stale KV（未接受的 d）由下一步 anchor query 覆盖，
context 长度（kv_len-1）排除 stale 区，故安全。

用法：
  CUDA_VISIBLE_DEVICES=3 python3 benchmark/validate_spec_decode.py \
      --model /models/Qwen3-0.6B --N 3 --max-new 64
"""
import argparse
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# RoPE（half-split / rotate_half，与 Qwen3 一致）
# ---------------------------------------------------------------------------
def build_rope_cache(head_dim, max_pos, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, :head_dim // 2].to(dtype)
    sin = emb.sin()[:, :head_dim // 2].to(dtype)
    return cos, sin


def rope_half_split(x, cos, sin):
    """x [..., d]，cos/sin [..., 1, d//2]。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# 目标模型 forward（手动 dense KV cache，causal / non-causal 可选）
# ---------------------------------------------------------------------------
class TargetRunner:
    """Qwen3 目标模型 forward，手动管理 dense KV cache。

    KV cache: k/v [num_layers, max_len, num_kv_heads, head_dim]（预分配 max_len）。
    forward(input_ids [T], positions [T], ctx_len, causal) -> logits [T, vocab]
      - ctx_len: 已 cache 的 context 长度（input_ids 之前的 token 数，读 KV[0:ctx_len]）。
      - input_ids 的 KV 写入 cache[ctx_len:ctx_len+T]。
      - causal: True=因果（verify），False=非因果（draft，mask token 互相可见）。
    """

    def __init__(self, model, device, dtype, max_len=4096):
        self.model = model
        self.device = device
        self.dtype = dtype
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.vocab_size = cfg.vocab_size
        self.max_len = max_len
        theta = getattr(cfg, "rope_theta", None) or (cfg.rope_parameters or {}).get("rope_theta", 1000000.0)
        self.cos, self.sin = build_rope_cache(self.head_dim, max_len, theta, device, dtype)
        self.k_cache = torch.zeros(self.num_layers, max_len, self.num_kv_heads, self.head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, input_ids, positions, ctx_len, causal=True):
        """input_ids [T]，positions [T]（绝对位置），ctx_len=读 context 长度。
        返回 logits [T, vocab]。input_ids 的 KV 写入 cache[ctx_len:ctx_len+T]。"""
        T = input_ids.shape[0]
        h = self.model.model.embed_tokens(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        end = ctx_len + T

        for li in range(self.num_layers):
            layer = self.model.model.layers[li]
            attn = layer.self_attn
            residual = h
            h_normed = layer.input_layernorm(h)
            q = attn.q_proj(h_normed).view(T, self.num_heads, self.head_dim)
            k = attn.k_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q = rope_half_split(q, cos, sin)
            k = rope_half_split(k, cos, sin)
            self.k_cache[li, ctx_len:end] = k
            self.v_cache[li, ctx_len:end] = v
            attn_out = self._attention(li, q, ctx_len, T, causal)
            attn_out = attn.o_proj(attn_out.reshape(T, -1))
            h = attn_out + residual
            residual = h
            h_normed = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            mlp_out = mlp.down_proj(F.silu(mlp.gate_proj(h_normed)) * mlp.up_proj(h_normed))
            h = mlp_out + residual

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def _attention(self, li, q, ctx_len, T, causal):
        """q [T, H, D]。KV 从 cache 读 [0:end]。GQA：kv 头 repeat 到 q 头。"""
        end = ctx_len + T
        k = self.k_cache[li, :end]
        v = self.v_cache[li, :end]
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.einsum("thd,ehd->hte", q, k) * (self.head_dim ** -0.5)
        if causal:
            q_pos = ctx_len + torch.arange(T, device=q.device)
            kv_pos = torch.arange(end, device=q.device)
            mask = kv_pos[None, :] <= q_pos[:, None]
            scores = scores.masked_fill(~mask[None], float("-inf"))
        attn = scores.softmax(-1)
        return torch.einsum("hte,ehd->thd", attn, v)


# ---------------------------------------------------------------------------
# 草稿模型（自起草：复用目标模型权重，非因果 forward，context 读目标 KV cache）
# ---------------------------------------------------------------------------
class DraftRunner:
    """自起草：用目标模型权重做非因果 forward（mask token 互相可见）。

    context KV 直接读目标模型的 KV cache（自起草时草稿=目标，KV 相同）。
    草稿无需独立持久 KV cache：mask token 的 KV 仅当前步 draft attention 用（互相可见），
    不持久化（下一步 context 是有效前缀，不含上一步的 mask）。
    """

    def __init__(self, model, target, device, dtype, max_len=4096, mask_token_id=0):
        self.model = model
        self.target = target  # 读其 KV cache 作 context
        self.device = device
        self.dtype = dtype
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.max_len = max_len
        self.mask_token_id = mask_token_id
        theta = getattr(cfg, "rope_theta", None) or (cfg.rope_parameters or {}).get("rope_theta", 1000000.0)
        self.cos, self.sin = build_rope_cache(self.head_dim, max_len, theta, device, dtype)

    def draft(self, anchor_token, ctx_len, N):
        """起草 N 个 token。

        anchor_token: 上一步 bonus token（int）。ctx_len: 读 context 长度（= kv_len-1，
        anchor 之前的 token 数）。N: 起草 token 数。
        喂 [anchor, mask*N]（1+N token，非因果），context=目标 KV[0:ctx_len]。
        返回 draft_tokens [N]（N 个 mask 位置的 argmax）。
        """
        T = 1 + N
        input_ids = torch.tensor([anchor_token] + [self.mask_token_id] * N,
                                 dtype=torch.long, device=self.device)
        positions = torch.arange(ctx_len, ctx_len + T, device=self.device).long()
        logits = self._forward_noncausal(input_ids, positions, ctx_len)
        return logits[1:].argmax(dim=-1)  # [N]（N 个 mask 位置）

    def _forward_noncausal(self, input_ids, positions, ctx_len):
        T = input_ids.shape[0]
        h = self.model.model.embed_tokens(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        end = ctx_len + T
        # 临时 KV buffer 存本步 1+N query 的 KV（context 读目标 cache）
        k_q = torch.empty(T, self.num_kv_heads, self.head_dim, dtype=self.dtype, device=self.device)
        v_q = torch.empty_like(k_q)
        for li in range(self.num_layers):
            layer = self.model.model.layers[li]
            attn = layer.self_attn
            residual = h
            h_normed = layer.input_layernorm(h)
            q = attn.q_proj(h_normed).view(T, self.num_heads, self.head_dim)
            k = attn.k_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q = rope_half_split(q, cos, sin)
            k = rope_half_split(k, cos, sin)
            k_q, v_q = k, v
            attn_out = self._attention_noncausal(li, q, ctx_len, T, k_q, v_q)
            attn_out = attn.o_proj(attn_out.reshape(T, -1))
            h = attn_out + residual
            residual = h
            h_normed = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            mlp_out = mlp.down_proj(F.silu(mlp.gate_proj(h_normed)) * mlp.up_proj(h_normed))
            h = mlp_out + residual
        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def _attention_noncausal(self, li, q, ctx_len, T, k_q, v_q):
        """非因果：query 看全部 context（目标 KV[0:ctx_len]）+ 全部 1+N query（互相可见）。"""
        k_ctx = self.target.k_cache[li, :ctx_len]
        v_ctx = self.target.v_cache[li, :ctx_len]
        k = torch.cat([k_ctx, k_q], dim=0)  # [ctx_len+T, KV, D]
        v = torch.cat([v_ctx, v_q], dim=0)
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.einsum("thd,ehd->hte", q, k) * (self.head_dim ** -0.5)
        attn = scores.softmax(-1)
        return torch.einsum("hte,ehd->thd", attn, v)


# ---------------------------------------------------------------------------
# Plain greedy（参考）
# ---------------------------------------------------------------------------
def plain_greedy(target, prompt_ids, max_new, device):
    target.reset()
    positions = torch.arange(len(prompt_ids), device=device).long()
    logits = target.forward(torch.tensor(prompt_ids, device=device), positions, 0, causal=True)
    tokens = list(prompt_ids)
    next_tok = int(logits[-1].argmax())
    tokens.append(next_tok)
    kv_len = len(prompt_ids)  # prompt KV 已 cache
    for _ in range(max_new - 1):
        pos = torch.tensor([kv_len], device=device).long()
        logits = target.forward(torch.tensor([next_tok], device=device), pos, kv_len, causal=True)
        next_tok = int(logits[-1].argmax())
        tokens.append(next_tok)
        kv_len += 1
    return tokens


# ---------------------------------------------------------------------------
# 投机解码（自起草，greedy 确定性接受）
# ---------------------------------------------------------------------------
def spec_decode(target, draft, prompt_ids, max_new, N, device, mask_token_id=0):
    """返回 (tokens, accepted_list)。"""
    target.reset()
    # prefill（目标 cache prompt KV；草稿无独立 cache，context 读目标）
    positions = torch.arange(len(prompt_ids), device=device).long()
    prompt_t = torch.tensor(prompt_ids, device=device)
    target.forward(prompt_t, positions, 0, causal=True)

    tokens = list(prompt_ids)
    kv_len = len(prompt_ids)  # prompt KV 已 cache
    anchor = prompt_ids[-1]
    accepted_list = []
    generated = 0
    while generated < max_new:
        remain = max_new - generated
        n_draft = min(N, remain)
        # 1. Draft：context = 目标 KV[0:kv_len-1]（anchor 之前）
        d = draft.draft(anchor, kv_len - 1, n_draft)  # [n_draft]
        d_list = [int(x) for x in d.tolist()]
        # 2. Verify：喂 [anchor, d_0..d_{n-1}]（1+n_draft），context = KV[0:kv_len-1]
        verify_ids = [anchor] + d_list
        verify_t = torch.tensor(verify_ids, device=device)
        verify_pos = torch.arange(kv_len - 1, kv_len - 1 + len(verify_ids), device=device).long()
        vlogits = target.forward(verify_t, verify_pos, kv_len - 1, causal=True)
        # target_preds[i] = p_{kv_len+i}（i=0..n_draft，共 n_draft+1 个）
        target_preds = [int(vlogits[i].argmax()) for i in range(len(verify_ids))]
        # 3. Accept
        accepted = 0
        for i in range(n_draft):
            if d_list[i] == target_preds[i]:
                accepted += 1
            else:
                break
        bonus = target_preds[accepted]
        new_tokens = d_list[:accepted] + [bonus]
        tokens.extend(new_tokens)
        generated += len(new_tokens)
        accepted_list.append(accepted)
        # 4. 更新：verify 写了 [anchor, d_0..d_{n-1}] 的 KV 到 [kv_len-1, kv_len-1+1+n_draft)。
        #    有效序列增长 len(new_tokens)=accepted+1（d_0..d_{accepted-1} + bonus）。
        #    kv_len 跟踪有效序列长度 L（ctx_len=kv_len-1 传给下一步 draft/verify）。
        #    bonus 的 KV 未写，由下一步 anchor query 补写（覆盖 stale 区首位置）。
        kv_len += accepted + 1
        anchor = bonus
    return tokens, accepted_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-0.6B")
    ap.add_argument("--N", type=int, default=3, help="投机 token 数")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is, and the reason is that")
    ap.add_argument("--mask-token", type=int, default=0)
    args = ap.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(0)
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")

    target = TargetRunner(model, device, dtype, max_len=4096)
    draft = DraftRunner(model, target, device, dtype, max_len=4096, mask_token_id=args.mask_token)

    # 1. Plain greedy（参考）
    t0 = time.time()
    ref_tokens = plain_greedy(target, prompt_ids, args.max_new, device)
    t_ref = time.time() - t0
    ref_out = ref_tokens[len(prompt_ids):]

    # 2. 投机解码
    t0 = time.time()
    spec_tokens, accepted_list = spec_decode(target, draft, prompt_ids, args.max_new,
                                             args.N, device, args.mask_token)
    t_spec = time.time() - t0
    spec_out = spec_tokens[len(prompt_ids):]

    # 3. 对比
    match = ref_out == spec_out
    print("\n" + "=" * 60)
    print(f"Plain greedy  : {len(ref_out)} tokens, {t_ref*1000:.1f}ms, {len(ref_out)/t_ref:.1f} tok/s")
    print(f"Spec decode   : {len(spec_out)} tokens, {t_spec*1000:.1f}ms, {len(spec_out)/t_spec:.1f} tok/s")
    print(f"Speedup       : {t_ref/t_spec:.2f}x")
    print(f"Output match  : {match}")
    if not match:
        for i, (a, b) in enumerate(zip(ref_out, spec_out)):
            if a != b:
                print(f"  首个分歧 @gen[{i}]: ref={a} spec={b}")
                print(f"  ref  : {tokenizer.decode(ref_out[:i+5])!r}")
                print(f"  spec : {tokenizer.decode(spec_out[:i+5])!r}")
                break
    avg_accept = sum(accepted_list) / len(accepted_list) if accepted_list else 0
    print(f"平均接受 token 数: {avg_accept:.3f} / N={args.N}（acceptance length = {avg_accept+1:.3f} 含 bonus）")
    print(f"接受分布(前20步): {accepted_list[:20]}")
    print("=" * 60)
    print(f"参考输出: {tokenizer.decode(ref_out)!r}")
    print(f"投机输出: {tokenizer.decode(spec_out)!r}")

    assert match, "投机解码输出与 plain greedy 不一致！"
    print("\n✅ 机制正确性验证通过：投机解码输出与无投机 greedy 逐 token 一致")


if __name__ == "__main__":
    main()
