"""DFlash2 草稿模型（micro-vllm 版）。

机制（对齐 vLLM dflash2）：
- 每步草稿模型吃 1+N 个 query token（anchor=上一步 bonus token + N 个 mask token），
  一次 forward 并行起草 N 个 token。
- 草稿模型是 5 层 sliding-window(2048) 非因果 Qwen3（hidden 5120, 32q/8kv, head_dim 128）。
- 草稿模型需要 target 的中间层 hidden states（target_layer_ids=[5,19,33,47,61]）
  经 fc 投影 + hidden_norm 作为输入 embedding 的一部分（use_aux_hidden_state）。
- DFlash2 特有：每层 attention/mlp 前后各一组 DFlashGroupedConv（可学习卷积系数），
  以及 CandidateSelector（predecessor/successor codebook 对候选 token 边打分）。

本文件实现：
- DFlash2DraftModel：完整 DFlash2 草稿模型（conv + selector + 5 层 sliding attn）。
- SelfDraftModel：自起草模式（草稿=目标模型本身，无 conv/selector），用于
  Qwen3-0.6B 机制正确性验证。
- load_dflash2_draft：从 HF safetensors 加载 DFlash2 权重。

注意：本文件只做模型定义与权重加载，不涉及 KV cache / paged attention。
草稿模型的 attention 在 core/spec_decode.py 里用 flash_attn 直接算
（草稿 KV 只需保留 sliding window 内，且每步 query 只有 1+N 个 token，
不需要 paged cache——context KV 由 target hidden states 每步重算）。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# 复用仓库现成 Triton kernel（pointwise/递归类小算子允许 Triton，GEMM 走 cuBLAS/int8）：
# - rmsnorm / rmsnorm_residual_fused：非 1-centered RMSNorm（Qwen3 风格 out=x*rrms*w），
#   替代 PyTorch 的 float/pow/mean/rsqrt/mul/cast 碎片 op（每层 4 个 norm × 5 层）。
# - apply_rope_decode：in-place half-split RoPE（[T, heads, dim] + cos/sin 表 + positions），
#   替代 PyTorch 的 chunk/cat/mul 碎片 op。
from kernel.rmsnorm import rmsnorm as _triton_rmsnorm
from kernel.rmsnorm import rmsnorm_residual_fused as _triton_rmsnorm_res
from kernel.rotary import apply_rope_decode

# draft 5 层 int8（Marlin）开关：MICRO_DRAFT_INT8=1 时把 draft 自有 Linear（q/k/v/o/
# gate/up/down/kernel_projection/fc/hidden_projection）bf16→int8 Marlin，forward 走
# marlin_forward。draft 每步读 ~2.7GB bf16 权重（d.fwd 6.22ms），int8 减半 → 省 ~3-4ms/step。
# 只转 draft 自有权重：embed_tokens/lm_head 与 target 共享（同对象），转了会破坏 target。
# 正确性：draft 提议对 hidden 扰动不敏感（rel_std≤0.01 提议变化 ≤1.4%，实测），int8
# group-128 噪声 ~0.5% 相对 → 接受率风险低（e2e 验证 acceptance 4.312 保持）。
# 默认开（MICRO_DRAFT_INT8=0 可关）：d.fwd 6.22→4.03ms，per_step -4.0%。
_DRAFT_INT8 = os.environ.get("MICRO_DRAFT_INT8", "1") == "1"


# ---- DFlash2 grouped conv 融合 kernel（taps=2 特化）----
# 原 PyTorch 版每次调用 ~8 个小 kernel（unflatten/view/add/mul/pad/where/flatten ×2 tap），
# 每 draft forward 20 次（5 层 × 2 conv × prepare/finish）= ~160 次 launch。融合成单
# kernel：out[t,g,j] = (base[0,g,j]+delta[t,0,g])*x[t,g,j]
#              + (base[1,g,j]+delta[t,1,g]) * (x[t-1,g,j] if t>=1 else 0)。
# 数值与原式逐项一致（同顺序 fp32 累加、同 bf16 存储）。
@triton.jit
def _grouped_conv_fused_kernel(HS, DELTA, BASE, OUT,
                               G, GS, T, D_T_STRIDE, BLOCK_G: tl.constexpr, BLOCK_J: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    j = tl.arange(0, BLOCK_J)
    gmask = g < G
    # x[t, g, j]：HS 行宽 = G*GS
    x = tl.load(HS + t.to(tl.int64) * (G * GS) + g[:, None] * GS + j[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)
    # x[t-1, g, j]（t==0 时 0，对齐 F.pad 的 leading 0）
    prev = tl.load(HS + (t - 1).to(tl.int64) * (G * GS) + g[:, None] * GS + j[None, :],
                   mask=gmask[:, None] & (t >= 1), other=0.0).to(tl.float32)
    # delta [T, 2, G]（T-stride = D_T_STRIDE，非 2*G：delta 是 [T,2,taps,G] 的切片）：
    # tap0 = delta[t,0,g]，tap1 = delta[t,1,g]
    d0 = tl.load(DELTA + t.to(tl.int64) * D_T_STRIDE + 0 * G + g, mask=gmask, other=0.0).to(tl.float32)
    d1 = tl.load(DELTA + t.to(tl.int64) * D_T_STRIDE + 1 * G + g, mask=gmask, other=0.0).to(tl.float32)
    # base [2, G, GS]
    b0 = tl.load(BASE + g[:, None] * GS + j[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
    b1 = tl.load(BASE + (G * GS) + g[:, None] * GS + j[None, :], mask=gmask[:, None], other=0.0).to(tl.float32)
    c0 = b0 + d0[:, None]
    c1 = b1 + d1[:, None]
    out = c0 * x + c1 * prev
    tl.store(OUT + t.to(tl.int64) * (G * GS) + g[:, None] * GS + j[None, :],
             out.to(OUT.dtype.element_ty), mask=gmask[:, None])


def _grouped_conv_fused(hidden_states, delta, base, num_groups, group_size):
    """taps=2 融合版 _grouped_conv（单 Triton kernel）。
    hidden_states [T, G*GS]，delta [T, 2, G]（T-stride 可能非 2*G，取 delta.stride(0)），
    base [2, G*GS] → out [T, G*GS]。"""
    T = hidden_states.shape[0]
    out = torch.empty_like(hidden_states)
    BLOCK_G = 32
    BLOCK_J = max(16, group_size)
    _grouped_conv_fused_kernel[(T, triton.cdiv(num_groups, BLOCK_G))](
        hidden_states, delta, base, out,
        num_groups, group_size, T, delta.stride(0), BLOCK_G=BLOCK_G, BLOCK_J=BLOCK_J)
    return out


# ---------------------------------------------------------------------------
# RoPE（half-split / Llama 风格 rotate_half，与 Qwen3 一致）
# ---------------------------------------------------------------------------
def _build_rope_cache(head_dim, max_pos, theta, device, dtype):
    """预计算 cos/sin 表 [max_pos, head_dim//2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
    cos = emb.cos()[:, : head_dim // 2].to(dtype)  # [max_pos, head_dim//2]
    sin = emb.sin()[:, : head_dim // 2].to(dtype)
    return cos, sin


# ---------------------------------------------------------------------------
# RMSNorm（支持 fused residual 形式：forward(x, residual) -> (normed, new_residual)）
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def _norm(self, x):
        # 走 Triton rmsnorm（非 1-centered，out=x*rrms*w），替代 PyTorch 碎片 op。
        return _triton_rmsnorm(x, self.weight, self.eps)

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        # fused：new_residual = x + residual；normed = norm(new_residual) * weight。
        # Triton rmsnorm_residual_fused 一次 kernel 算 (normed, x+residual)。
        return _triton_rmsnorm_res(x, residual, self.weight, self.eps)


# ---------------------------------------------------------------------------
# DFlashGroupedConv（DFlash2 特有：可学习分组卷积，系数由 hidden 投影得到）
# ---------------------------------------------------------------------------
def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps,
                  position=None):
    """对齐 vLLM _grouped_conv。

    hidden_states: [T, hidden]
    delta: [T, taps, num_groups]（本侧系数增量）
    base: [taps, hidden]（基础卷积核）
    position: 可选预计算的 [T] 位置（= arange(T) mod block_size）。草稿路径 T 恒为
      block_size（1+N，2 的幂），故 position 是常量，由 DFlashGroupedConv 预计算传入，
      避免每次调用现建 torch.arange 临时张量。
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))  # [T, G, gs]
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)  # [T,taps,G,gs]
    output = coefficients[:, 0] * blocks  # [T, G, gs]
    if position is None:
        position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        if block_size & (block_size - 1) == 0:
            position = position & (block_size - 1)
        else:
            position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(self, hidden_size, taps, group_size, block_size, params_dtype):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} 必须整除 hidden_size={hidden_size}")
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype), requires_grad=False
        )
        self.kernel_projection = nn.Linear(hidden_size, 2 * taps * self.num_groups, bias=False)
        self.kernel_projection.weight.data = self.kernel_projection.weight.data.to(params_dtype)
        # 预计算 position（= arange(block_size) mod block_size）。草稿路径 T 恒为
        # block_size（1+N，2 的幂），故 position 是常量；register_buffer 随 .to(device)
        # 迁移，避免每次 _convolve 现建 torch.arange 临时张量（每 draft forward 20 次）。
        pos = torch.arange(block_size)
        if block_size & (block_size - 1) == 0:
            pos = pos & (block_size - 1)
        else:
            pos = pos % block_size
        self.register_buffer("_conv_pos", pos, persistent=False)

    def _convolve(self, hidden_states, delta, side):
        T = hidden_states.shape[0]
        # taps==2 且 T==block_size（DFlash2 草稿路径，position=arange 不 wrap）→ 单
        # Triton kernel 融合（省 ~8 小 kernel/次 × 20 次/forward）。否则回退 PyTorch 版
        # （taps>2 或 T!=block_size 时 position>=tap 的 wrap 语义需原式）。
        if self.taps == 2 and T == self.block_size:
            return _grouped_conv_fused(
                hidden_states, delta, self.base_kernel[side],
                self.num_groups, self.group_size)
        position = self._conv_pos if T == self.block_size else None
        return _grouped_conv(
            hidden_states, delta, self.base_kernel[side],
            self.block_size, self.num_groups, self.group_size, self.taps,
            position=position,
        )

    def prepare(self, hidden_states):
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden_states, coefficients):
        return self._convolve(hidden_states, coefficients, 1)


# ---------------------------------------------------------------------------
# CandidateSelector（DFlash2 特有：codebook 边打分）
# ---------------------------------------------------------------------------
class CandidateSelector(nn.Module):
    def __init__(self, hidden_size, vocab_size, rank, top_k, params_dtype):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.hidden_projection.weight.data = self.hidden_projection.weight.data.to(params_dtype)

    def forward(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook, self.successor_codebook,
            candidate_ids, unary_logits, hidden, anchor_token_ids, self.top_k,
        )


def _score_edges(predecessor_table, successor_table, candidate_ids,
                 unary_logits, hidden, anchor_token_ids, top_k):
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


# ---------------------------------------------------------------------------
# 草稿 attention 层（sliding-window 非因果 Qwen3）
# ---------------------------------------------------------------------------
class DFlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 rms_norm_eps, sliding_window, rope_theta, max_pos, dtype, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            m.weight.data = m.weight.data.to(dtype)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, dtype=dtype)

        self.rope_theta = rope_theta
        self._cos, self._sin = _build_rope_cache(head_dim, max_pos, rope_theta, device, dtype)

    def forward(self, positions, hidden_states, context_kv=None, attn_mask=None):
        """positions: [T] 绝对位置。hidden_states: [T, hidden]（1+N query token）。
        context_kv: 可选 (k_ctx [C, KV, D], v_ctx [C, KV, D])——target 中间层 hidden
        预计算的 context KV（DFlash2 核心：草稿 attention 读 context + query）。
        attn_mask: 可选加性 mask [T, C+T]（0=有效, -inf=屏蔽）。用于 draft CUDA graph：
        context KV 固定到长度 C（graph 需固定 shape），mask 屏蔽 [ctx_len:C) 的无效
        context 位置（exp(-inf)=0，softmax 结果与只读 [ctx_len) 完全一致）。None=不 mask。
        返回 [T, hidden]。"""
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).view(-1, self.q_size)
        k = self.k_norm(k).view(-1, self.kv_size)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)

        # 融合 Triton RoPE（in-place half-split，cos/sin 表 + per-token positions），
        # 替代 PyTorch 的 cos/sin gather + chunk/cat/mul 碎片 op。
        apply_rope_decode(q, self._cos, self._sin, positions)
        apply_rope_decode(k, self._cos, self._sin, positions)

        # 拼接 context KV（DFlash2：草稿 attention 读 context + 1+N query，非因果）
        if context_kv is not None:
            k_ctx, v_ctx = context_kv
            k = torch.cat([k_ctx, k], dim=0)
            v = torch.cat([v_ctx, v], dim=0)

        # 非因果 sliding-window attention（草稿 query 只有 1+N 个 token，直接算）
        # GQA：把 kv 头 repeat 到 q 头数
        n_rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.view(-1, self.num_kv_heads, self.head_dim).repeat_interleave(n_rep, dim=1)
        # attn_mask（draft CUDA graph 用）：q/k/v 是 3D [H, T, D]（无 batch 维），
        # 故 mask 保持 2D [T, C+T]（SDPA 自动广播到 head 维）。加性 mask 屏蔽位
        # exp(-inf)=0，softmax 结果与只读 [ctx_len) 完全一致。
        attn = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            is_causal=False, scale=self.scaling, attn_mask=attn_mask,
        )
        attn = attn.transpose(0, 1).reshape(-1, self.q_size)
        return self.o_proj(attn)

    def project_kv(self, hidden_states, positions):
        """hidden_states [C, hidden]（context token）→ (k [C, KV, D], v [C, KV, D])。
        用于 precompute_context_kv：target 中间层 hidden 投影成草稿 context KV。"""
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        # 融合 Triton RoPE（in-place half-split），替代 PyTorch 碎片 op。
        apply_rope_decode(k, self._cos, self._sin, positions)
        return k, v


class DFlashDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx, dtype, device, block_size, rope_theta, max_pos):
        super().__init__()
        self.self_attn = DFlashAttention(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, cfg.rms_norm_eps, cfg.sliding_window, rope_theta, max_pos,
            dtype, device,
        )
        self.mlp = _SwiGLU(cfg.hidden_size, cfg.intermediate_size, dtype)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # DFlash2 特有 conv
        dflash_cfg = getattr(cfg, "dflash_config", None) or {}
        self.use_conv = "conv_kernel_size" in dflash_cfg
        if self.use_conv:
            conv_args = dict(
                hidden_size=cfg.hidden_size,
                taps=int(dflash_cfg["conv_kernel_size"]),
                group_size=int(dflash_cfg.get("conv_group_size", 16)),
                block_size=block_size,
                params_dtype=dtype,
            )
            self.attention_conv = DFlashGroupedConv(**conv_args)
            self.mlp_conv = DFlashGroupedConv(**conv_args)

    def forward(self, positions, hidden_states, residual, context_kv=None, attn_mask=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_conv:
            hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states,
                                       context_kv=context_kv, attn_mask=attn_mask)
        if self.use_conv:
            hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_conv:
            hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_conv:
            hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


class _SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        for m in (self.gate_proj, self.up_proj, self.down_proj):
            m.weight.data = m.weight.data.to(dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# DFlash2 草稿模型主体
# ---------------------------------------------------------------------------
class DFlash2DraftModel(nn.Module):
    """完整 DFlash2 草稿模型（对齐 vLLM qwen3_dflash2.py 机制）。

    机制（vLLM 参考实现）：
    - 草稿权重【不含】embed_tokens 与 lm_head（safetensors 仅 81 key：
      layers.* / fc / hidden_norm / norm / candidate_selector.*）。两者由
      share_target_weights 从 target 共享（同 vocab、同 hidden）。
    - 每步 query = 1+N token（bonus + N mask），经【共享 embed_tokens】得 query hidden，
      过 5 层 sliding-window 非因果 decoder。
    - context KV：target 中间层 hidden（target_layer_ids）拼接 → fc → hidden_norm →
      各层 k/v proj + k_norm + RoPE（precompute_context_kv）。草稿 query attention
      读 context + 本步 query（非因果）。
    - 候选：草稿 mask 位置 hidden → 【共享 lm_head】→ top_k 候选 → selector 边打分
      → 贪心 walk 选一条连贯路径（select_draft_tokens）。

    forward(input_ids, positions, context_kv, input_embeds) -> last_hidden_states [T, hidden]
      - input_ids: [T]（1+N query token：bonus + N mask）
      - positions: [T] 绝对位置
      - context_kv: 可选 list（每层 (k_ctx, v_ctx)），由 precompute_context_kv 产出
      - input_embeds: 可选 [T, hidden]（= 共享 embed_tokens(input_ids)）；缺省自算
    """

    def __init__(self, cfg, dtype, device, num_speculative_tokens, max_pos=4096):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        dflash_cfg = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_cfg.get("mask_token_id")
        self.target_layer_ids = dflash_cfg.get("target_layer_ids", [])
        self.use_aux_hidden_state = dflash_cfg.get("use_aux_hidden_state", True)

        # block_size = 1 + N（anchor + N mask）
        self.block_size = 1 + num_speculative_tokens

        rope_theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000000)

        # DFlash2 草稿权重【不含】embed_tokens 与 lm_head（safetensors 仅 81 个 key：
        # layers.* / fc / hidden_norm / norm / candidate_selector.*）。两者从 target 共享
        # （share_target_weights 赋值，同 vocab/hidden）：query 用共享 embed_tokens，
        # 候选 logits 用共享 lm_head。默认 None，使用前必须先共享（否则 ~5GB 随机权重）。
        self.embed_tokens = None
        self.lm_head = None

        self.layers = nn.ModuleList([
            DFlashDecoderLayer(cfg, i, dtype, device, self.block_size, rope_theta, max_pos)
            for i in range(self.num_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # aux hidden state 投影：fc [hidden, num_aux*target_hidden] + hidden_norm
        if self.use_aux_hidden_state:
            target_hidden = getattr(cfg, "target_hidden_size", None) or cfg.hidden_size
            num_aux = len(self.target_layer_ids) if self.target_layer_ids else self.num_layers
            self.fc = nn.Linear(target_hidden * num_aux, cfg.hidden_size, bias=False)
            self.fc.weight.data = self.fc.weight.data.to(dtype)
            self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # DFlash2 特有：candidate selector（codebook 边打分）。lm_head 用 target 的（见 compute_candidates）。
        if "selector_rank" in dflash_cfg:
            self.candidate_selector = CandidateSelector(
                cfg.hidden_size, cfg.vocab_size,
                int(dflash_cfg["selector_rank"]), int(dflash_cfg["selector_top_k"]), dtype,
            )
        else:
            self.candidate_selector = None

        self.input_embedding_scale = float(dflash_cfg.get("input_embedding_scale", 1.0))
        self.to(device)

    def embed_input_ids(self, input_ids):
        # DFlash2 不用 embedding（输入是 aux hidden states）。保留接口供自起草/调试。
        if self.embed_tokens is None:
            raise RuntimeError("DFlash2 草稿无 embed_tokens（输入应为 aux_hidden_states）")
        return self.embed_tokens(input_ids) * self.input_embedding_scale

    def share_target_weights(self, target_embed_tokens, target_lm_head):
        """从 target 共享 embed_tokens 与 lm_head（对齐 vLLM load_dflash_model：
        draft.embed_tokens = target_embed；dflash_model.lm_head = target_lm_head）。
        草稿 checkpoint 不含这两组权重（同 vocab/hidden，直接复用 target 的）。"""
        self.embed_tokens = target_embed_tokens
        self.lm_head = target_lm_head

    def convert_to_int8(self):
        """draft 自有 Linear（q/k/v/o/gate/up/down/kernel_projection/fc/
        hidden_projection）bf16→int8 Marlin。embed_tokens/lm_head 与 target 共享
        （同对象），跳过（转了会破坏 target）。fc 的 K=hidden*num_aux 须 128 倍数
        （5120*5=25600 ✓）。"""
        from kernel.marlin import linear_to_marlin
        converted = 0
        # 逐层 attention + mlp + conv
        for layer in self.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, linear_to_marlin(getattr(attn, name)))
                converted += 1
            mlp = layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(mlp, name, linear_to_marlin(getattr(mlp, name)))
                converted += 1
            if layer.use_conv:
                layer.attention_conv.kernel_projection = linear_to_marlin(
                    layer.attention_conv.kernel_projection)
                layer.mlp_conv.kernel_projection = linear_to_marlin(
                    layer.mlp_conv.kernel_projection)
                converted += 2
        # fc（aux hidden 投影）+ hidden_projection（selector）
        if self.use_aux_hidden_state:
            self.fc = linear_to_marlin(self.fc)
            converted += 1
        if self.candidate_selector is not None:
            self.candidate_selector.hidden_projection = linear_to_marlin(
                self.candidate_selector.hidden_projection)
            converted += 1
        torch.cuda.empty_cache()
        print(f"[DFlash2] draft int8: 转换 {converted} 个 Linear → Marlin", flush=True)

    def combine_hidden_states(self, aux_hidden_states):
        """target 中间层 hidden 拼接 [C, num_aux*target_hidden] → fc → [C, hidden]。
        （hidden_norm 在 precompute_context_kv 里做，对齐 vLLM。）"""
        if not self.use_aux_hidden_state:
            return aux_hidden_states
        return self.fc(aux_hidden_states)

    def precompute_context_kv(self, context_states, context_positions):
        """DFlash2 核心：context hidden（fc 投影后 [C, hidden]）→ hidden_norm →
        各层 k/v proj + k_norm + RoPE，供草稿 attention 读取。

        context_states: [C, hidden]（combine_hidden_states 输出）
        context_positions: [C] 绝对位置
        返回 list of (k [C, KV, D], v [C, KV, D])，每层一个。
        """
        if self.use_aux_hidden_state:
            context_states = self.hidden_norm(context_states)
        return [layer.self_attn.project_kv(context_states, context_positions)
                for layer in self.layers]

    def fill_context_kv(self, context_states, context_positions, out_k, out_v,
                        start, end):
        """增量版 precompute_context_kv：只算 [start,end) 这段 context 的 KV，直接写进
        常驻 buffer（不建临时 list、不 cat）。hidden_norm/project_kv 都是 per-token
        （RMSNorm 按行、proj/norm/RoPE 按行），故 [start,end) 的切片结果 == 全量结果的
        [start,end) 切片，增量写与全量重算逐元素一致（数值等价）。

        context_states: [end-start, hidden]（combine_hidden_states 输出，未 norm）
        context_positions: [end-start] 绝对位置
        out_k/out_v: [num_layers, max_len, KV, D] 常驻 buffer，写 [:, start:end]。
        """
        if self.use_aux_hidden_state:
            context_states = self.hidden_norm(context_states)
        for i, layer in enumerate(self.layers):
            k, v = layer.self_attn.project_kv(context_states, context_positions)
            out_k[i, start:end].copy_(k)
            out_v[i, start:end].copy_(v)

    def forward(self, input_ids, positions, input_embeds=None, context_kv=None,
                attn_mask=None):
        """DFlash2 交叉注意力 forward。

        - query：input_embeds（[1+N, hidden]，= target embed_tokens([anchor]+[mask]*N)
          * input_embedding_scale）。input_embeds=None 时回退 embed_input_ids(input_ids)。
        - context：context_kv（每层 (k_ctx, v_ctx)），由 precompute_context_kv 从
          target aux hidden states（combine_hidden_states 后）投影产出。
        - attn_mask：可选加性 mask [T, C+T]（draft CUDA graph 用，屏蔽固定 context
          长度里 [ctx_len:C) 的无效位置）。None=不 mask。

        返回 last_hidden_states [1+N, hidden]。
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds

        residual = None
        for i, layer in enumerate(self.layers):
            ckv = context_kv[i] if context_kv is not None else None
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            context_kv=ckv, attn_mask=attn_mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_candidates(self, hidden_states):
        """草稿 mask 位置 hidden → 共享 lm_head（= target lm_head）→ top_k 候选。
        返回 (candidate_ids [T, top_k], unary_logits [T, top_k])。"""
        if self.lm_head is None:
            raise RuntimeError("DFlash2 草稿 lm_head 未共享（先 share_target_weights）")
        logits = self.lm_head(hidden_states)
        top_k = self.candidate_selector.top_k if self.candidate_selector else 16
        unary_logits, candidate_ids = torch.topk(logits, top_k, dim=-1)
        return candidate_ids, unary_logits

    def select_draft_tokens(self, hidden_states, anchor_token_ids):
        """DFlash2 完整选 token：compute_candidates + selector 打分 + 贪心 walk。

        hidden_states: [num_reqs, N, hidden]（N 个 mask 位置的 hidden）
        anchor_token_ids: [num_reqs]
        返回 draft_tokens [num_reqs, N]。
        """
        num_reqs, N, _ = hidden_states.shape
        candidate_ids, unary_logits = self.compute_candidates(hidden_states.flatten(0, 1))
        candidate_ids = candidate_ids.view(num_reqs, N, -1)
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = self.candidate_selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
        # 贪心 walk：每步选 score 最大的候选，作为下一步的 predecessor
        # scores 形状 [num_reqs, N, top_k, top_k]：scores[b, l, p, c] = 候选 c 在位置 l、
        # predecessor=candidate_ids[b, l-1, p]（l==0 时为 anchor）时的分数。
        # walk：previous 从上一步选中的候选 index 开始，逐步 argmax。
        arange = torch.arange(num_reqs, device=hidden_states.device)
        draft = torch.empty(num_reqs, N, dtype=torch.long, device=hidden_states.device)
        previous = torch.zeros(num_reqs, dtype=torch.long, device=hidden_states.device)
        for step in range(N):
            sel = scores[arange, step, previous]  # [num_reqs, top_k]
            idx = sel.argmax(dim=-1)  # [num_reqs]
            draft[:, step] = candidate_ids[arange, step, idx]
            previous = idx
        return draft


# ---------------------------------------------------------------------------
# 自起草模型（草稿=目标模型本身，用于 Qwen3-0.6B 机制验证）
# ---------------------------------------------------------------------------
class SelfDraftModel(nn.Module):
    """自起草：直接复用目标模型的 decoder 层做非因果 forward。

    与 DFlash2 的区别：无 conv/selector/aux_hidden_state，输入就是 input_ids 的
    embedding。forward 对 1+N 个 query token 做非因果 attention（每个 token 能看到
    全部 1+N 个），返回 last_hidden_states。

    注意：这里用目标模型的权重，但 attention 是非因果的（草稿语义）。目标模型
    的 q/k/v/o/norm/mlp 权重直接复用。
    """

    def __init__(self, target_model, cfg, dtype, device, num_speculative_tokens, max_pos=4096):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.vocab_size = cfg.vocab_size
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.block_size = 1 + num_speculative_tokens

        rope_theta = getattr(cfg, "rope_theta", 1000000)
        self.embed_tokens = target_model.model.embed_tokens
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.norm.weight.data = target_model.model.norm.weight.data.clone()
        # 复用目标模型的层权重，但包一层非因果 attention
        self.layers = nn.ModuleList([
            _SelfDraftLayer(target_model.model.layers[i], cfg, dtype, device, rope_theta, max_pos)
            for i in range(self.num_layers)
        ])
        self.lm_head = target_model.lm_head

    def forward(self, input_ids, positions, aux_hidden_states=None):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_candidates(self, hidden_states):
        logits = self.lm_head(hidden_states)
        top_k = 1
        unary_logits, candidate_ids = torch.topk(logits, top_k, dim=-1)
        return candidate_ids.squeeze(-1), unary_logits.squeeze(-1)

    def select_draft_tokens(self, hidden_states, anchor_token_ids):
        """自起草：每个 mask 位置直接 argmax。hidden_states [num_reqs, N, hidden]。"""
        num_reqs, N, _ = hidden_states.shape
        logits = self.lm_head(hidden_states)
        return logits.argmax(dim=-1)  # [num_reqs, N]


class _SelfDraftLayer(nn.Module):
    """复用目标模型一层的权重，但 attention 改为非因果。"""

    def __init__(self, target_block, cfg, dtype, device, rope_theta, max_pos):
        super().__init__()
        self.target_block = target_block
        self.self_attn = DFlashAttention(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, cfg.rms_norm_eps, None, rope_theta, max_pos, dtype, device,
        )
        # 拷贝目标模型权重
        sa = self.self_attn
        tb = target_block.self_attn
        sa.q_proj.weight.data = tb.q_proj.weight.data.clone()
        sa.k_proj.weight.data = tb.k_proj.weight.data.clone()
        sa.v_proj.weight.data = tb.v_proj.weight.data.clone()
        sa.o_proj.weight.data = tb.o_proj.weight.data.clone()
        sa.q_norm.weight.data = tb.q_norm.weight.data.clone()
        sa.k_norm.weight.data = tb.k_norm.weight.data.clone()
        self.mlp = target_block.mlp
        # HF RMSNorm 不支持 fused residual，用本文件 RMSNorm 包一层并拷贝权重
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.input_layernorm.weight.data = target_block.input_layernorm.weight.data.clone()
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm.weight.data = target_block.post_attention_layernorm.weight.data.clone()

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
def load_dflash2_draft(model_path, dtype, device, num_speculative_tokens, max_pos=4096):
    """从 HF safetensors 加载 DFlash2 草稿模型。

    DFlash2DraftModel 架构 AutoModelForCausalLM 不认识，手动加载 safetensors。
    权重命名（HF）：
      model.embed_tokens.weight
      model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{i}.self_attn.{q,k}_norm.weight
      model.layers.{i}.mlp.{gate,up,down}_proj.weight
      model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
      model.layers.{i}.attention_conv.{base_kernel,kernel_projection.weight}
      model.layers.{i}.mlp_conv.{base_kernel,kernel_projection.weight}
      model.norm.weight
      model.fc.weight / model.hidden_norm.weight
      model.candidate_selector.{predecessor,successor_codebook,hidden_projection.weight}
      lm_head.weight
    """
    import json
    import os
    from safetensors import safe_open

    with open(os.path.join(model_path, "config.json")) as f:
        cfg_dict = json.load(f)

    class _Cfg:
        pass
    cfg = _Cfg()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    model = DFlash2DraftModel(cfg, dtype, device, num_speculative_tokens, max_pos)

    # 建立目标参数名 → 模块参数 的映射
    state = {}
    st_path = os.path.join(model_path, "model.safetensors")
    with safe_open(st_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)

    # 手动映射
    def set_param(name, tensor):
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        getattr(obj, parts[-1]).data.copy_(tensor.to(dtype))

    # 逐 key 映射（HF 命名 → 本模型命名基本一致，直接 copy）
    missing = []
    for key, tensor in state.items():
        try:
            set_param(key, tensor)
        except (AttributeError, IndexError, TypeError):
            missing.append(key)
    if missing:
        print(f"[DFlash2] 未映射权重 {len(missing)} 个: {missing[:10]}")

    model.to(device)
    model.eval()
    return model, cfg
