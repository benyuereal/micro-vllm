import torch


class Sampler:
    def __init__(self):
        # 编译采样函数（静态形状）
        self._compiled_sample = torch.compile(
            self._sample_impl,
            fullgraph=True,
            dynamic=True,  # batch_size 固定
            mode="reduce-overhead",

            # mode="max-autotune"
        )

    def __call__(self, logits, temperatures, top_ps, top_k,
                 prev_tokens=None, rep_penalties=None,
                 all_greedy=None, any_rep_pen=None):
        """调用采样函数。temperature<=0 走 greedy（argmax），绕过编译路径避免 0 除。

        prev_tokens: [bs, max_prev] 已生成+prompt 的 token id（含 padding，-1 表无效），
                     用于 repetition penalty。None 或 rep_penalties 全 1.0 时跳过惩罚。
        rep_penalties: [bs] 每条 seq 的惩罚系数（1.0=禁用，>1.0 惩罚已出现 token）。
        all_greedy: CPU bool，调用方预判 temperatures 全 <=0（避免 torch.any GPU→CPU 同步）。
        any_rep_pen: CPU bool，调用方预判 rep_penalties 存在 >1.0（避免 torch.any 同步）。
        """
        # repetition penalty（在 greedy/采样前统一施加，纯 GPU op，graph-friendly）
        # 优先用 CPU 侧预判标志避免 torch.any 同步；无标志时回退 GPU 检查。
        if any_rep_pen is not None:
            need_pen = any_rep_pen
        else:
            need_pen = (prev_tokens is not None and rep_penalties is not None
                        and bool(torch.any(rep_penalties > 1.0)))
        if need_pen:
            logits = self._apply_repetition_penalty(logits, prev_tokens, rep_penalties)

        # greedy 短路：任一行 temperature<=0 即走 eager argmax（与 HF do_sample=False 对齐）
        if all_greedy is not None:
            is_greedy = all_greedy
        else:
            is_greedy = bool(torch.any(temperatures <= 0))
        if is_greedy:
            return logits.argmax(dim=-1)
        return self._compiled_sample(logits, temperatures, top_ps, top_k)

    @staticmethod
    def _apply_repetition_penalty(logits, prev_tokens, rep_penalties):
        """HF 约定：对历史出现过的 token，logit>0 则 /p，logit<0 则 *p（p>1 降低概率）。
        prev_tokens: [bs, L]（-1 padding）。rep_penalties: [bs]。纯 GPU，无 host 同步。"""
        # 构造 [bs, vocab] 出现掩码：只在 valid（非 -1 padding）位置 scatter 1.0。
        # 用 valid 作 scatter 的 value，padding 位置写 0 → 不污染。
        valid = (prev_tokens >= 0).to(logits.dtype)      # [bs, L]
        safe_idx = prev_tokens.clamp(min=0)              # 避免负索引
        mask = torch.zeros_like(logits)                  # [bs, vocab]
        mask.scatter_(1, safe_idx, valid)                # 出现=1，padding=0
        pen = rep_penalties[:, None]                      # [bs, 1]
        penalized = torch.where(logits > 0, logits / pen, logits * pen)
        return torch.where(mask > 0, penalized, logits)

    @staticmethod
    def _sample_impl(logits, temp, top_p, top_k):
        # 整个采样在一个 fused kernel 内
        logits = logits / temp[:, None]

        # Top-K
        vals, idxs = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(vals, dim=-1)

        # Top-P 过滤（简化版：直接取 cumsum < top_p）
        sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
        cum_p = torch.cumsum(sorted_p, dim=-1)
        valid = cum_p < top_p[:, None]
        valid[..., 0] = True  # 至少保留一个

        probs_masked = sorted_p * valid
        probs_norm = probs_masked / probs_masked.sum(dim=-1, keepdim=True)

        # 采样
        samples = torch.multinomial(probs_norm, 1)

        # 映射
        topk_idx = sorted_i.gather(-1, samples)
        return idxs.gather(-1, topk_idx).squeeze(-1)
