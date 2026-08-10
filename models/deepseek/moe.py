"""
DeepSeek MoE prefill 路径（eager，按 expert 分段批算）。

decode 路径已由 kernel/moe.py 的 TileLang 融合 kernel（moe_decode_tilelang）接管，
本模块只保留 prefill 用的大 batch 分段实现。

📌 设计：
    - gate: hidden → [n_experts] softmax → top-k (greedy, n_group=1 走简单 topk 分支)
    - routed experts: 64 个 expert MLP（gate/up/down SwiGLU），按 expert 分段批算
    - shared experts: 2 个共享专家合并成单个大 MLP（intermediate = moe_intermediate * n_shared）
    - norm_topk_prob=false + routed_scaling_factor=1.0 → topk_weight 不归一化、不缩放

🔑 数据流（prefill, x: [B*S, hidden]）：
    1. logits = x @ gate_weight.T           # [N, n_experts]
    2. scores = softmax(logits, fp32)
    3. topk_weight, topk_idx = topk(scores, k)   # [N, k]
    4. 按 expert 分段：sorted by topk_idx → 每段跑对应 expert → scatter 回 → ×topk_weight → sum
    5. + shared_experts(x)
"""
import torch
import torch.nn.functional as F


def moe_forward(x, gate_weight, e_gu, e_d, top_k, n_experts,
                shared_gu=None, shared_d=None, decode=False):
    """prefill MoE 前向（按 expert 分段批算，合并同 expert 多 token）。

    x: [N, hidden]  (N = B*S for prefill)
    gate_weight: [n_experts, hidden]
    e_gu: [n_experts, 2*inter, hidden]  (fused gate|up, 转置好用于 x @ gu.T)
    e_d:  [n_experts, hidden, inter]
    decode: 仅保留接口兼容性；True 时回退到下面的分段路径（prefill 风格，对小 N 亦正确）。
    返回: [N, hidden]
    """
    N = x.shape[0]
    hidden = x.shape[1]

    # 1. gating（fp32 计算 scores，与 HF 一致）
    logits = F.linear(x, gate_weight)              # [N, E]
    scores = logits.softmax(dim=-1, dtype=torch.float32).to(x.dtype)

    # 2. top-k (greedy: n_group=1 → 纯 topk)
    topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    # norm_topk_prob=false, routed_scaling_factor=1.0 → 不归一化、不缩放

    # 3. expert SwiGLU 计算：按 expert 分段批算
    flat_idx = topk_idx.reshape(-1)                # [N*k]
    flat_w = topk_weight.reshape(-1)               # [N*k]

    x_rep = x.unsqueeze(1).expand(N, top_k, hidden).reshape(N * top_k, hidden)
    order = flat_idx.argsort()                  # 按 expert 分组排序
    sorted_idx = flat_idx[order]
    sorted_x = x_rep[order]
    sorted_w = flat_w[order]
    counts = torch.bincount(sorted_idx, minlength=n_experts)  # [E]
    out_rep = torch.empty_like(sorted_x)        # [N*k, hidden]
    cum = 0
    counts_list = counts.tolist()
    for ei, cnt in enumerate(counts_list):
        if cnt == 0:
            continue
        seg = sorted_x[cum:cum + cnt]           # [cnt, hidden]
        gu = e_gu[ei]                           # [2*inter, hidden]
        d = e_d[ei]                             # [hidden, inter]
        gate_up = seg @ gu.t()                  # [cnt, 2*inter]
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        out_rep[cum:cum + cnt] = act @ d.t()
        cum += cnt
    inv_order = order.argsort()
    out_rep = out_rep[inv_order]
    out = (out_rep.view(N, top_k, hidden) *
           sorted_w[inv_order].view(N, top_k, 1).to(out_rep.dtype)).sum(dim=1)

    # 4. shared experts (shared_gu: [hidden, 2*s_inter] = [gate|up].t(), shared_d: [s_inter, hidden])
    if shared_gu is not None:
        gate_up = x @ shared_gu              # [N, 2*s_inter] = [gate_out | up_out]
        gate, up = gate_up.chunk(2, dim=-1)  # [N, s_inter]
        out = out + (F.silu(gate) * up) @ shared_d  # [N, hidden]

    return out
