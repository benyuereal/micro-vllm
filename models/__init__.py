"""
models 包 - 按模型架构分派适配器。

用法:
    from models import build_adapter
    adapter = build_adapter(model.config)   # 返回 QwenAdapter / DeepSeekAdapter ...
"""
from .base import ModelAdapter


def build_adapter(cfg) -> ModelAdapter:
    """根据 config.model_type (或 architectures) 选择适配器。"""
    model_type = getattr(cfg, "model_type", "").lower()
    archs = [a.lower() for a in getattr(cfg, "architectures", []) or []]

    # DeepSeek-V2 / V3 系列 (MLA + MoE)
    if model_type.startswith("deepseek_v2") or model_type == "deepseek_v3" \
            or any("deepseekv2" in a or "deepseekv3" in a for a in archs):
        from .deepseek.adapter import DeepSeekAdapter
        return DeepSeekAdapter()

    # Qwen3.5 (GDN 线性注意力 + full attention 混合，1-centered RMSNorm)。
    # 必须先于 Qwen3 判断：arch "qwen3_5forconditionalgeneration" 含子串 "qwen3"。
    if model_type == "qwen3_5" or model_type == "qwen3_5_text" \
            or any("qwen3_5" in a for a in archs):
        from .qwen3_5.adapter import Qwen3_5Adapter
        return Qwen3_5Adapter()

    # Qwen3 (GQA + SwiGLU + QK-Norm，HF 命名，head_dim 独立)
    if model_type == "qwen3" or any("qwen3" in a for a in archs):
        from .qwen3.adapter import Qwen3Adapter
        return Qwen3Adapter()

    # Qwen-1 / Qwen2 / Qwen2.5 (GQA + SwiGLU) —— 默认/兜底
    from .qwen.adapter import QwenAdapter
    return QwenAdapter()


__all__ = ["ModelAdapter", "build_adapter"]
