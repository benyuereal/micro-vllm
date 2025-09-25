# File: weight_rearrange.py
import torch


def rearrange_qwen_weights(model):
    """
    Qwen7B 权重重排优化
    """
    for layer in model.layers:
        # 重排 QKV 权重
        if hasattr(layer.attn, "c_attn"):
            weight = layer.attn.c_attn.weight
            if hasattr(weight, "qweight"):
                weight.qweight.data = _rearrange(weight.qweight.data)
            else:
                weight.data = _rearrange(weight.data)

        # 重排输出投影权重
        if hasattr(layer.attn, "c_proj"):
            weight = layer.attn.c_proj.weight
            if hasattr(weight, "qweight"):
                weight.qweight.data = _rearrange(weight.qweight.data)
            else:
                weight.data = _rearrange(weight.data)

    return model


def _rearrange(weight: torch.Tensor) -> torch.Tensor:
    """
    权重重排核心函数
    """
    out_features, in_features = weight.shape
    group_size = 128
    num_groups = (in_features + group_size - 1) // group_size

    rearranged = torch.empty_like(weight)

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, in_features)
        group_weight = weight[:, start_idx:end_idx]

        # 按 warp 大小重排
        for warp_idx in range(0, group_weight.shape[1], 32):
            warp_end = min(warp_idx + 32, group_weight.shape[1])
            warp = group_weight[:, warp_idx:warp_end]

            new_start = start_idx + warp_idx
            new_end = start_idx + warp_end
            rearranged[:, new_start:new_end] = warp

    return rearranged