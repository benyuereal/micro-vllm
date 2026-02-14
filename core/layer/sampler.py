import torch


class Sampler:
    def __init__(self):
        # 编译采样函数（静态形状）
        self._compiled_sample = torch.compile(
            self._sample_impl,
            fullgraph=True,
            dynamic=False,  # batch_size 固定
            mode="reduce-overhead",

            # mode="max-autotune"
        )
    
    def __call__(self, logits, temperatures, top_ps, top_k):
        """调用编译后的采样函数"""
        return self._compiled_sample(logits, temperatures, top_ps, top_k)
    
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