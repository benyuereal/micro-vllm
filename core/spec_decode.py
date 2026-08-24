"""DFlash2 投机解码控制器（engine 集成）。

机制（对齐 vLLM dflash2，已在 benchmark/validate_spec_decode.py 独立验证）：
- 每步草稿模型用 1+N 个 query token（anchor + N 个 mask token）并行起草 N 个 token，
  target 模型一次 forward（1+N token，因果）验证，greedy 下确定性接受
  （draft token 与 target argmax 一致则接受，遇首个不一致停止，bonus=target 预测）。
- 正确性：接受的 token 全是 target 自己的预测，故投机解码输出与无投机 greedy 逐 token 一致。

本控制器自包含（dense KV cache + 自管 forward），复用 engine 的模型权重与 tokenizer，
与 engine 的 paged cache 解耦。用于：
  1. 机制正确性端到端验证（Qwen3-0.6B 自起草 / oracle）。
  2. 单用户 decode 吞吐对比（有/无投机解码）。
W8A16 目标模型就绪后，可迁移到 paged cache + CUDA graph 路径以获取真实加速。

权重布局兼容：
- prepared（engine 的 Qwen3Adapter.prepare_weights 后）：attn._qkv_w/_o_w/_q_norm_w/...，
  mlp._gu/_d，block._in_ln_w/_post_ln_w。
- HF 原始（transformers 加载，未 prepare）：attn.q_proj/k_proj/...，mlp.gate_proj/...。
  两种布局用同一 forward（_layer_forward 内按 hasattr 分派）。

索引约定（与验证脚本一致）：
- kv_len = 有效序列长度 L。anchor = tokens[L-1]。
- Draft/Verify 的 context = KV[0:L-1]（anchor 之前），写入 [anchor, d_0..d_{N-1}] 的
  KV 到 [L-1, L+N)。接受 accepted 个 draft + bonus 后，kv_len += accepted+1。
- 未接受的 draft KV 是 stale，由下一步 anchor query 覆盖，context 长度排除 stale 区。
"""
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE（half-split / rotate_half，与 Qwen3 一致）
# ---------------------------------------------------------------------------
def _build_rope_cache(head_dim, max_pos, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, :head_dim // 2].to(dtype)
    sin = emb.sin()[:, :head_dim // 2].to(dtype)
    return cos, sin


def _rope_half_split(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _rmsnorm_head(x, weight, eps):
    """per-head RMSNorm：x [T, H, D]，weight [D]。fp32 计算，返回 x.dtype。"""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf * weight.float()).to(dtype)


def _rmsnorm_full(x, weight, eps):
    """full RMSNorm：x [T, H]，weight [H]。"""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf * weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# 投机解码控制器
# ---------------------------------------------------------------------------
class SpecDecodeController:
    """DFlash2 投机解码控制器（单序列，dense KV cache）。

    复用 engine 的模型权重（target_model）做 verify，草稿模型（DFlash2 或自起草）
    独立 forward。draft 与 target 可同模型（自起草）或不同（DFlash2 小草稿 + 大目标）。
    """

    def __init__(self, target_model, draft_model, device, dtype,
                 num_speculative_tokens: int = 7, mask_token_id: int = 0,
                 max_len: int = 4096, draft_is_target: bool = False):
        self.target = target_model
        self.draft = draft_model
        self.device = device
        self.dtype = dtype
        self.N = num_speculative_tokens
        self.mask_token_id = mask_token_id
        self.max_len = max_len
        self.draft_is_target = draft_is_target

        cfg = target_model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.vocab_size = cfg.vocab_size
        theta = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1000000.0)
        self.cos, self.sin = _build_rope_cache(self.head_dim, max_len, theta, device, dtype)

        # target 的 dense KV cache
        self.k_cache = torch.zeros(self.num_layers, max_len, self.num_kv_heads, self.head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)

        # 统计
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

    # ---------------- 单层 forward（兼容 prepared / HF 权重布局） ----------------
    def _layer_forward(self, model, li, h, cos, sin, ctx_len, T, causal):
        """单层：input_layernorm → attention → o_proj → residual → post_ln → mlp → residual。

        causal=True（verify）：KV 写 self.k_cache[ctx_len:ctx_len+T]，attention 读 [0:ctx_len+T]。
        causal=False（draft）：attention 读 [target KV[0:ctx_len] + 本步 query KV]
        （非因果，mask 互相可见），不写持久 KV。
        返回新 h [T, hidden]。
        """
        layer = model.model.layers[li]
        attn = layer.self_attn
        mlp = layer.mlp
        prepared = hasattr(attn, "_qkv_w")

        # input_layernorm
        if prepared:
            h_normed = _rmsnorm_full(h, layer._in_ln_w, layer._in_ln_eps)
        else:
            h_normed = layer.input_layernorm(h)

        # QKV 投影
        if prepared:
            qkv = h_normed @ attn._qkv_w.t()  # [T, qkv_dim]
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_kv_heads * self.head_dim
            q = qkv[:, :q_dim].view(T, self.num_heads, self.head_dim)
            k = qkv[:, q_dim:q_dim + kv_dim].view(T, self.num_kv_heads, self.head_dim)
            v = qkv[:, q_dim + kv_dim:].view(T, self.num_kv_heads, self.head_dim)
            q = _rmsnorm_head(q, attn._q_norm_w, attn._q_norm_eps)
            k = _rmsnorm_head(k, attn._k_norm_w, attn._k_norm_eps)
        else:
            q = attn.q_proj(h_normed).view(T, self.num_heads, self.head_dim)
            k = attn.k_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(h_normed).view(T, self.num_kv_heads, self.head_dim)
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        # RoPE
        q = _rope_half_split(q, cos, sin)
        k = _rope_half_split(k, cos, sin)

        # attention
        if causal:
            self.k_cache[li, ctx_len:ctx_len + T] = k
            self.v_cache[li, ctx_len:ctx_len + T] = v
            attn_out = self._attention(li, q, ctx_len, T, causal=True)
        else:
            attn_out = self._draft_attention(li, q, ctx_len, T, k, v)

        # o_proj
        if prepared:
            attn_out = attn_out.reshape(T, -1) @ attn._o_w.t()
        else:
            attn_out = attn.o_proj(attn_out.reshape(T, -1))

        h = attn_out + h
        residual = h

        # post_attention_layernorm
        if prepared:
            h_normed = _rmsnorm_full(h, layer._post_ln_w, layer._post_ln_eps)
        else:
            h_normed = layer.post_attention_layernorm(h)

        # mlp (SwiGLU)
        if prepared:
            gu = h_normed @ mlp._gu.t()  # [T, 2*inter]，前半 up、后半 gate
            inter = gu.shape[1] // 2
            up = gu[:, :inter]
            gate = gu[:, inter:]
            h = (F.silu(gate) * up) @ mlp._d.t()
        else:
            h = mlp.down_proj(F.silu(mlp.gate_proj(h_normed)) * mlp.up_proj(h_normed))

        return h + residual

    def _attention(self, li, q, ctx_len, T, causal):
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

    def _draft_attention(self, li, q, ctx_len, T, k_q, v_q):
        """非因果：query 看全部 context（target KV[0:ctx_len]）+ 全部 1+N query。"""
        k_ctx = self.k_cache[li, :ctx_len]
        v_ctx = self.v_cache[li, :ctx_len]
        k = torch.cat([k_ctx, k_q], dim=0)
        v = torch.cat([v_ctx, v_q], dim=0)
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.einsum("thd,ehd->hte", q, k) * (self.head_dim ** -0.5)
        attn = scores.softmax(-1)
        return torch.einsum("hte,ehd->thd", attn, v)

    # ---------------- 模型 forward ----------------
    def _forward(self, model, input_ids, positions, ctx_len, causal):
        """完整 forward。返回 logits [T, vocab]。

        causal=True（verify）：写 target KV cache。
        causal=False（draft）：不写持久 KV（mask KV 仅本步用）。
        """
        T = input_ids.shape[0]
        h = model.model.embed_tokens(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        for li in range(self.num_layers):
            h = self._layer_forward(model, li, h, cos, sin, ctx_len, T, causal)
        h = model.model.norm(h)
        return model.lm_head(h)

    # ---------------- 主接口 ----------------
    def prefill(self, prompt_ids: List[int]):
        """prefill prompt，返回首 token（target 预测）。"""
        self.reset()
        positions = torch.arange(len(prompt_ids), device=self.device).long()
        logits = self._forward(self.target, torch.tensor(prompt_ids, device=self.device),
                               positions, 0, causal=True)
        return int(logits[-1].argmax())

    def step(self, tokens: List[int], kv_len: int, anchor: int, n_draft: int):
        """一步投机解码。返回 (new_tokens, accepted)。

        tokens: 当前有效 token 序列（长度 >= kv_len）。kv_len: 有效序列长度 L。
        anchor: tokens[L-1]。n_draft: 本步起草 token 数（<= N）。
        """
        # 1. Draft（非因果，context = target KV[0:kv_len-1]）
        draft_model = self.target if self.draft_is_target else self.draft
        input_ids = torch.tensor([anchor] + [self.mask_token_id] * n_draft,
                                 dtype=torch.long, device=self.device)
        positions = torch.arange(kv_len - 1, kv_len - 1 + 1 + n_draft, device=self.device).long()
        dlogits = self._forward(draft_model, input_ids, positions, kv_len - 1, causal=False)
        d_list = [int(x) for x in dlogits[1:].argmax(dim=-1).tolist()]

        # 2. Verify（因果，写 target KV cache）
        verify_ids = [anchor] + d_list
        verify_t = torch.tensor(verify_ids, device=self.device)
        verify_pos = torch.arange(kv_len - 1, kv_len - 1 + len(verify_ids), device=self.device).long()
        vlogits = self._forward(self.target, verify_t, verify_pos, kv_len - 1, causal=True)
        target_preds = [int(vlogits[i].argmax()) for i in range(len(verify_ids))]

        # 3. Accept（greedy 确定性）
        accepted = 0
        for i in range(n_draft):
            if d_list[i] == target_preds[i]:
                accepted += 1
            else:
                break
        bonus = target_preds[accepted]
        new_tokens = d_list[:accepted] + [bonus]

        self.total_accepted += accepted
        self.total_steps += 1
        self.total_generated += len(new_tokens)
        return new_tokens, accepted

    @property
    def avg_acceptance(self) -> float:
        return self.total_accepted / self.total_steps if self.total_steps else 0.0

    def generate(self, prompt_ids: List[int], max_new_tokens: int,
                 eos_token_id: Optional[int] = None) -> List[int]:
        """完整生成。返回新生成的 token 列表。"""
        tokens = list(prompt_ids)
        kv_len = len(prompt_ids)
        anchor = self.prefill(prompt_ids)
        tokens.append(anchor)
        kv_len += 1
        generated = 1
        while generated < max_new_tokens:
            if eos_token_id is not None and anchor == eos_token_id:
                break
            n_draft = min(self.N, max_new_tokens - generated)
            new_tokens, _ = self.step(tokens, kv_len, anchor, n_draft)
            tokens.extend(new_tokens)
            kv_len += len(new_tokens)
            generated += len(new_tokens)
            anchor = new_tokens[-1]
        return tokens[len(prompt_ids):]
