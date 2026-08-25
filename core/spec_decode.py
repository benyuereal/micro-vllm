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

权重布局：HF 原始（transformers 加载，未 prepare）——attn.q_proj/k_proj/...，
mlp.gate_proj/...。engine 的 _build_spec_controller 加载的正是新鲜未 prepare 副本
（prepare_weights 会重排权重，无法用于本控制器的逐层 forward）。

公共算子（RoPE / RMSNorm）复用 kernel/dflash_ops.py 与 kernel/rmsnorm.py，
attention 用 flash_attn（原生 GQA，无 repeat_interleave）。

索引约定（与验证脚本一致）：
- kv_len = 有效序列长度 L。anchor = tokens[L-1]。
- Draft/Verify 的 context = KV[0:L-1]（anchor 之前），写入 [anchor, d_0..d_{N-1}] 的
  KV 到 [L-1, L+N)。接受 accepted 个 draft + bonus 后，kv_len += accepted+1。
- 未接受的 draft KV 是 stale，由下一步 anchor query 覆盖，context 长度排除 stale 区。
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

from kernel.dflash_ops import build_rope_cache, rope_half_split
from kernel.rmsnorm import rmsnorm


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
        self.cos, self.sin = build_rope_cache(self.head_dim, max_len, theta, device, dtype)

        # target 的 dense KV cache
        self.k_cache = torch.zeros(self.num_layers, max_len, self.num_kv_heads, self.head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)

        # DFlash2 草稿：target 中间层 hidden states 提取（context 用）
        # target_layer_ids 来自草稿模型 config（DFlash2 才有）；自起草为空。
        draft_cfg = getattr(draft_model, "cfg", None)
        dflash_cfg = getattr(draft_cfg, "dflash_config", None) or {}
        self.target_layer_ids = list(dflash_cfg.get("target_layer_ids", []))
        self._aux_layer_set = set(self.target_layer_ids)
        # DFlash2 路径：草稿是独立 DFlash2DraftModel（有 precompute_context_kv）。
        # 自起草（draft_is_target）走旧的 target 非因果 forward 路径。
        self.use_dflash2 = (not draft_is_target) and hasattr(draft_model, "precompute_context_kv")
        self.hidden_size = cfg.hidden_size
        if self.use_dflash2:
            # aux_cache[ai, pos] = target 第 target_layer_ids[ai] 层在位置 pos 的 hidden。
            # 由 prefill/verify 的 _forward(collect_aux=True) 填充，供草稿建 context KV。
            self.aux_cache = torch.zeros(
                len(self.target_layer_ids), max_len, self.hidden_size,
                dtype=dtype, device=device)
        else:
            self.aux_cache = None
        # 草稿 sliding window（context 上限）；DFlash2=2048。
        self.draft_sliding_window = int(getattr(draft_cfg, "sliding_window", 0) or 0)

        # 统计
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        if self.aux_cache is not None:
            self.aux_cache.zero_()
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

    # ---------------- 单层 forward（HF 权重布局） ----------------
    def _layer_forward(self, model, li, h, cos, sin, ctx_len, T, causal, write_kv=True):
        """单层：input_layernorm → attention → o_proj → residual → post_ln → mlp → residual。

        causal=True（verify）：KV 写 self.k_cache[ctx_len:ctx_len+T]（write_kv=True 时），
        attention 读 [0:ctx_len+T]。
        causal=False（draft）：attention 读 [target KV[0:ctx_len] + 本步 query KV]
        （非因果，mask 互相可见），不写持久 KV。
        write_kv=False：不写 KV cache（用于提取 target aux hidden states，context KV 已在 cache）。
        返回新 h [T, hidden]。
        """
        layer = model.model.layers[li]
        attn = layer.self_attn
        mlp = layer.mlp
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        # input_layernorm
        h_normed = layer.input_layernorm(h)

        # QKV 投影 + per-head QK-Norm + RoPE
        q = attn.q_norm(attn.q_proj(h_normed).view(T, nh, hd))
        k = attn.k_norm(attn.k_proj(h_normed).view(T, nkv, hd))
        v = attn.v_proj(h_normed).view(T, nkv, hd)
        q = rope_half_split(q, cos, sin)
        k = rope_half_split(k, cos, sin)

        # attention（flash_attn 原生 GQA，无 repeat_interleave）
        if causal:
            if write_kv:
                self.k_cache[li, ctx_len:ctx_len + T] = k
                self.v_cache[li, ctx_len:ctx_len + T] = v
            attn_out = self._verify_attention(li, q, ctx_len, T)
        else:
            attn_out = self._draft_attention(li, q, ctx_len, T, k, v)

        # o_proj + residual
        h = attn.o_proj(attn_out.reshape(T, -1)) + h

        # post_attention_layernorm + mlp (SwiGLU) + residual
        h_normed = layer.post_attention_layernorm(h)
        return mlp.down_proj(F.silu(mlp.gate_proj(h_normed)) * mlp.up_proj(h_normed)) + h

    def _verify_attention(self, li, q, ctx_len, T):
        """因果 verify：query [ctx_len, ctx_len+T) 读 KV [0:ctx_len+T]（含本步写入）。
        varlen 单序列 causal（q 从 ctx_len 起，kv 从 0 起，等价于原 mask 逻辑）。"""
        end = ctx_len + T
        k = self.k_cache[li, :end].unsqueeze(0)  # [1, end, nkv, hd]
        v = self.v_cache[li, :end].unsqueeze(0)
        cu_q = torch.tensor([0, T], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, end], dtype=torch.int32, device=q.device)
        out = flash_attn_varlen_func(
            q, k.squeeze(0), v.squeeze(0), cu_q, cu_k, T, end,
            softmax_scale=self.head_dim ** -0.5, causal=True)
        return out  # [T, nh, hd]

    def _draft_attention(self, li, q, ctx_len, T, k_q, v_q):
        """非因果：query 看全部 context（target KV[0:ctx_len]）+ 全部 1+N query。"""
        k = torch.cat([self.k_cache[li, :ctx_len], k_q], dim=0).unsqueeze(0)  # [1, S, nkv, hd]
        v = torch.cat([self.v_cache[li, :ctx_len], v_q], dim=0).unsqueeze(0)
        out = flash_attn_func(
            q.unsqueeze(0), k, v,
            softmax_scale=self.head_dim ** -0.5, causal=False)
        return out.squeeze(0)  # [T, nh, hd]

    # ---------------- 模型 forward ----------------
    def _forward(self, model, input_ids, positions, ctx_len, causal,
                 write_kv=True, collect_aux=False):
        """完整 forward。返回 logits [T, vocab]（collect_aux=True 时返回 (logits, aux)）。

        causal=True（verify）：写 target KV cache（write_kv=True 时）。
        causal=False（draft）：不写持久 KV（mask KV 仅本步用）。
        collect_aux=True：额外收集 target_layer_ids 各层的 hidden states（DFlash2 草稿
        context 用），返回 (logits, aux [T, num_aux*hidden])。
        """
        T = input_ids.shape[0]
        h = model.model.embed_tokens(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        aux_parts = [] if collect_aux else None
        for li in range(self.num_layers):
            h = self._layer_forward(model, li, h, cos, sin, ctx_len, T, causal,
                                    write_kv=write_kv)
            if collect_aux and li in self._aux_layer_set:
                aux_parts.append(h)
        h = model.model.norm(h)
        logits = model.lm_head(h)
        if collect_aux:
            return logits, torch.cat(aux_parts, dim=-1)
        return logits

    def _extract_aux(self, input_ids, positions, ctx_len):
        """提取 target 中间层 hidden states（DFlash2 草稿 context 用）。

        返回 aux [T, num_aux*hidden]（target_layer_ids 各层 hidden 拼接）。
        不写 KV cache（context KV 已在 self.k_cache，由 prefill/verify 写入）。
        """
        _, aux = self._forward(self.target, input_ids, positions, ctx_len,
                               causal=True, write_kv=False, collect_aux=True)
        return aux

    # ---------------- 主接口 ----------------
    def prefill(self, prompt_ids: List[int]):
        """prefill prompt，返回首 token（target 预测）。"""
        self.reset()
        positions = torch.arange(len(prompt_ids), device=self.device).long()
        logits = self._forward(self.target, torch.tensor(prompt_ids, device=self.device),
                               positions, 0, causal=True)
        return int(logits[-1].argmax())

    def _draft_tokens(self, tokens: List[int], kv_len: int, anchor: int, n_draft: int) -> List[int]:
        """起草 n_draft 个 token。返回 d_list（长度 n_draft）。

        - 自起草（draft_is_target）：target 模型非因果 forward（context = target KV[0:kv_len-1]）。
        - DFlash2（独立草稿）：交叉注意力——context KV 由 target aux hidden states 投影，
          query = target embed_tokens([anchor]+[mask]*n_draft)，draft hidden 经 target lm_head。
        """
        if self.draft_is_target:
            input_ids = torch.tensor([anchor] + [self.mask_token_id] * n_draft,
                                     dtype=torch.long, device=self.device)
            positions = torch.arange(kv_len - 1, kv_len - 1 + 1 + n_draft, device=self.device).long()
            dlogits = self._forward(self.target, input_ids, positions, kv_len - 1, causal=False)
            return [int(x) for x in dlogits[1:].argmax(dim=-1).tolist()]

        # ---- DFlash2 交叉注意力草稿 ----
        # context = tokens[0:kv_len-1]（anchor 之前）。target aux hidden states（不写 KV，
        # context KV 已在 cache）→ combine_hidden_states（fc+hidden_norm）→ 各层 context KV。
        ctx_len = kv_len - 1
        context_ids = torch.tensor(tokens[:ctx_len], dtype=torch.long, device=self.device)
        context_pos = torch.arange(ctx_len, device=self.device).long()
        aux = self._extract_aux(context_ids, context_pos, 0)  # [ctx_len, num_aux*hidden]
        context_states = self.draft.combine_hidden_states(aux)  # [ctx_len, hidden]
        context_kv = self.draft.precompute_context_kv(context_states, context_pos)

        # query = target embed_tokens([anchor]+[mask]*n_draft) * input_embedding_scale
        query_ids = torch.tensor([anchor] + [self.mask_token_id] * n_draft,
                                 dtype=torch.long, device=self.device)
        query_embeds = self.target.model.embed_tokens(query_ids) * self.draft.input_embedding_scale
        query_pos = torch.arange(kv_len - 1, kv_len - 1 + 1 + n_draft, device=self.device).long()
        draft_hidden = self.draft(query_ids, query_pos, input_embeds=query_embeds,
                                  context_kv=context_kv)  # [1+n_draft, hidden]
        # DFlash2 无自有 lm_head，用 target 的
        dlogits = self.target.lm_head(draft_hidden)
        return [int(x) for x in dlogits[1:].argmax(dim=-1).tolist()]

    def step(self, tokens: List[int], kv_len: int, anchor: int, n_draft: int):
        """一步投机解码。返回 (new_tokens, accepted)。

        tokens: 当前有效 token 序列（长度 >= kv_len）。kv_len: 有效序列长度 L。
        anchor: tokens[L-1]。n_draft: 本步起草 token 数（<= N）。
        """
        # 1. Draft
        d_list = self._draft_tokens(tokens, kv_len, anchor, n_draft)

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
