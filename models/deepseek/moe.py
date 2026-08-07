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

from kernel.grouped_gemv import grouped_gate_up, grouped_down


def moe_forward(x, gate_weight, e_gu, e_d, top_k, n_experts,
                shared_gu=None, shared_d=None, decode=False):
    """
    x: [N, hidden]  (N = bs for decode, 或 B*S for prefill)
    gate_weight: [n_experts, hidden]   (已 .T 备好或原始均可，内部处理)
    e_gu: [n_experts, 2*inter, hidden]  (fused gate|up, 转置好用于 x @ gu.T)
    e_d:  [n_experts, hidden, inter]
    decode: True=decode 路径（CUDA Graph 友好，逐 token grouped GEMV，无 .item()/.tolist() 同步）；
            False=prefill 路径（大 batch，按 expert 分段批算，合并同 expert 多 token）。
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

    # 3. expert SwiGLU 计算
    flat_idx = topk_idx.reshape(-1)                # [N*k]
    flat_w = topk_weight.reshape(-1)               # [N*k]

    if decode:
        # ---- decode 路径（任意 N，CUDA Graph 友好）----
        # 逐 token 调 grouped GEMV（kernel 内按 expert_idx 索引权重，无 gather、无 host 同步）。
        # Python for range(N) 在 capture 时按固定 N 静态展开，replay 单 graph；每 iter 纯 GPU op。
        # grouped_down 输出 [1, hidden] 是单 token 的 K expert 加权累加，故必须逐 token 独立调用。
        out = torch.empty(N, hidden, dtype=x.dtype, device=x.device)
        w_ones = torch.ones(top_k, dtype=x.dtype, device=x.device)
        for i in range(N):
            idx_i = flat_idx[i * top_k:(i + 1) * top_k].to(torch.int64)  # [K]
            w_i = flat_w[i * top_k:(i + 1) * top_k]                      # [K]
            gu = grouped_gate_up(x[i:i + 1], e_gu, idx_i)               # [K, 2*inter]
            gate, up = gu.chunk(2, dim=-1)                              # [K, inter] each
            act = F.silu(gate) * up * w_i.unsqueeze(-1).to(gu.dtype)    # [K, inter] 已含权重
            out[i:i + 1] = grouped_down(act, e_d, idx_i, w_ones)        # [1, hidden]
    else:
        # ---- prefill 路径: 按 expert 分段批算（合并同 expert 多 token）----
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
