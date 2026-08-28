"""DFlash2 投机解码草稿模型。

- DFlash2DraftModel：5 层 sliding-window 非因果 Qwen3 草稿模型（hidden 5120,
  32q/8kv, head_dim 128, sliding_window 2048），带 DFlash2 特有的
  DFlashGroupedConv（attention/mlp 前后各一组）+ CandidateSelector
  （predecessor/successor codebook 边打分）。
- 权重从 HF safetensors 直接加载（AutoModelForCausalLM 不认识 DFlash2DraftModel
  架构，需手动映射）。
"""
from .draft_model import DFlash2DraftModel, load_dflash2_draft

__all__ = ["DFlash2DraftModel", "load_dflash2_draft"]
