"""
DeepSeek MoE 推理（eager，正确性优先，CUDA-Graph 友好预留）。

📌 设计：
    - gate: hidden → [n_experts] softmax → top-k (greedy, n_group=1 走简单 topk 分支)
    - routed experts: 64 个 expert MLP（gate/up/down SwiGLU），按 expert 分段批算
    - shared experts: 2 个共享专家合并成单个大 MLP（intermediate = moe_intermediate * n_shared）
    - norm_topk_prob=false + routed_scaling_factor=1.0 → topk_weight 不归一化、不缩放

🔑 数据流（decode, x: [bs, hidden]）：
    1. logits = x @ gate_weight.T           # [bs, n_experts]
    2. scores = softmax(logits, fp32)
    3. topk_weight, topk_idx = topk(scores, k)   # [bs, k]
    4. 按 expert 分段：sorted by topk_idx → 每段跑对应 expert → scatter 回 → ×topk_weight → sum
    5. + shared_experts(x)

⚡ 实现：
    - 权重预堆叠：experts 的 gate/up 合并为 _e_gu [E, 2*inter, hidden]，
      down 合并为 _e_d [E, hidden, inter]，用 index_select + bmm 批量算。
    - 这是 data-dependent 路由，首版 eager（不进 CUDA Graph 静态捕获）。
"""
import torch
import torch.nn.functional as F


def moe_forward(x, gate_weight, e_gu, e_d, top_k, n_experts,
                shared_gu=None, shared_d=None):
    """
    x: [N, hidden]  (N = bs for decode, 或 B*S for prefill)
    gate_weight: [n_experts, hidden]   (已 .T 备好或原始均可，内部处理)
    e_gu: [n_experts, 2*inter, hidden]  (fused gate|up, 转置好用于 x @ gu.T)
    e_d:  [n_experts, hidden, inter]
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

    # 3. 按 expert 分段批算
    #    把每个 (token, expert) 对展开，按 expert 排序，连续段送对应 expert
    flat_idx = topk_idx.reshape(-1)                # [N*k]
    flat_w = topk_weight.reshape(-1)               # [N*k]
    # 每个 token 重复 k 次得到对应 hidden
    x_rep = x.unsqueeze(1).expand(N, top_k, hidden).reshape(N * top_k, hidden)  # [N*k, hidden]

    order = flat_idx.argsort()                      # 按 expert 分组排序
    sorted_idx = flat_idx[order]
    sorted_x = x_rep[order]
    sorted_w = flat_w[order]

    # 各 expert 的 token 数
    counts = torch.bincount(sorted_idx, minlength=n_experts)  # [E]
    # 用 bmm 批量：对每个 expert 取其段算 SwiGLU。这里用循环段（专家数 64，段内 bmm）
    out_rep = torch.empty_like(sorted_x)            # [N*k, hidden]
    cum = 0
    counts_list = counts.tolist()
    for ei, cnt in enumerate(counts_list):
        if cnt == 0:
            continue
        seg = sorted_x[cum:cum + cnt]               # [cnt, hidden]
        gu = e_gu[ei]                               # [2*inter, hidden]
        d = e_d[ei]                                 # [hidden, inter]
        gate_up = seg @ gu.t()                      # [cnt, 2*inter] = [gate_out | up_out]
        gate, up = gate_up.chunk(2, dim=-1)         # gu=cat([gate_w, up_w]) → 首 gate, 次 up
        act = F.silu(gate) * up                     # DeepSeek 标准 SwiGLU: silu(gate)*up
        out_rep[cum:cum + cnt] = act @ d.t()        # [cnt, hidden]
        cum += cnt

    # scatter 回 (token, k) 顺序，加权求和
    inv_order = order.argsort()
    out_rep = out_rep[inv_order]                    # [N*k, hidden]
    out_rep = out_rep.view(N, top_k, hidden) * sorted_w[inv_order].view(N, top_k, 1).to(out_rep.dtype)
    out = out_rep.sum(dim=1)                        # [N, hidden]

    # 4. shared experts (shared_gu: [hidden, 2*s_inter] = [gate|up].t(), shared_d: [s_inter, hidden])
    if shared_gu is not None:
        gate_up = x @ shared_gu              # [N, 2*s_inter] = [gate_out | up_out]
        gate, up = gate_up.chunk(2, dim=-1)  # [N, s_inter]
        out = out + (F.silu(gate) * up) @ shared_d  # [N, hidden]

    return out
