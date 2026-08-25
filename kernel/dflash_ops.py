"""DFlash2 投机解码公共算子。

把散落在 core/spec_decode.py 与 models/dflash/draft_model.py 的重复实现收敛到一处：
- build_rope_cache / rope_half_split：RoPE（half-split / rotate_half，与 Qwen3 一致）。
  此前 spec_decode.py 与 draft_model.py 各有一份逐字相同的拷贝。
- grouped_conv：DFlash2 可学习分组卷积（系数 = base_kernel + hidden 投影增量）。
- score_edges：CandidateSelector 的 codebook 边打分（predecessor·hidden + successor）。

纯 torch 实现（小 op，T=1+N 或 C 行，无现成 TileLang/Triton kernel，ROI 低）。
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE（half-split / rotate_half，与 Qwen3 一致）
# ---------------------------------------------------------------------------
def build_rope_cache(head_dim, max_pos, theta, device, dtype):
    """预计算 cos/sin 表 [max_pos, head_dim//2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, :head_dim // 2].to(dtype)
    sin = emb.sin()[:, :head_dim // 2].to(dtype)
    return cos, sin


def rope_half_split(x, cos, sin):
    """half-split RoPE（Llama 风格 rotate_half）：x [..., d]，cos/sin [..., d//2]。
    q/k 共用。返回旋转后的 x（同形状）。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# DFlashGroupedConv 的核心卷积（可学习分组卷积）
# ---------------------------------------------------------------------------
def grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    """对齐 vLLM _grouped_conv。

    hidden_states: [T, hidden]
    delta: [T, taps, num_groups]（本侧系数增量）
    base: [taps, hidden]（基础卷积核）
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))  # [T, G, gs]
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)  # [T,taps,G,gs]
    output = coefficients[:, 0] * blocks  # [T, G, gs]
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


# ---------------------------------------------------------------------------
# CandidateSelector 的 codebook 边打分
# ---------------------------------------------------------------------------
def score_edges(predecessor_table, successor_table, candidate_ids,
                unary_logits, hidden, anchor_token_ids, top_k):
    """predecessor·hidden + successor 边打分。

    predecessor_table/successor_table: [vocab, rank]
    candidate_ids: [B, L]（anchor + 已选 candidate）
    unary_logits: [B, L, top_k]
    hidden: [B, L, rank]
    anchor_token_ids: [B]
    返回 [B, L, top_k]。
    """
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
