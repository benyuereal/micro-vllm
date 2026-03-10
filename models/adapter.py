import json
import torch
from pathlib import Path
from types import SimpleNamespace as NS


def load_std_model(hf_model, model_type: str) -> NS:
    """
    输入: HuggingFace模型, 模型类型(qwen/llama)
    输出: 一个命名通俗易懂的通用对象
    """
    # 1. 加载配置
    cfg = json.load(open(Path(__file__).parent / "configs" / f"{model_type}.json"))

    # 2. 递归取数工具
    def _get(obj, path):
        for p in path.split('.'): obj = getattr(obj, p)
        return obj

    # 3. 构建标准对象
    m = NS()
    m.config = hf_model.config
    m.embedding = _get(hf_model, cfg['embedding'])  # wte -> embedding
    m.final_norm = _get(hf_model, cfg['final_norm'])  # ln_f -> final_norm
    m.lm_head = _get(hf_model, cfg['lm_head'])  # lm_head 保持不变，业内通用
    m.blocks = []  # blocks 保持不变

    # 4. 遍历处理每个 block
    for hf_block in _get(hf_model, cfg['blocks']):
        b = NS()

        # --- Attention ---
        if 'qkv' in cfg['attn']:
            b.qkv = _get(hf_block, cfg['attn']['qkv'])
            b.qkv_bias = _get(hf_block, cfg['attn']['qkv_bias']) if cfg['attn']['qkv_bias'] else None
        else:
            q, k, v = [_get(hf_block, cfg['attn'][k]) for k in ['q', 'k', 'v']]
            b.qkv = torch.cat([q, k, v], dim=0)
            b.qkv_bias = None
        b.attn_out = _get(hf_block, cfg['attn']['out'])  # o -> attn_out

        # --- MLP ---
        gate, up = [_get(hf_block, cfg['mlp'][k]) for k in ['gate', 'up']]
        b.gate_up = torch.cat([gate, up], dim=0)  # gu -> gate_up
        b.mlp_down = _get(hf_block, cfg['mlp']['down'])  # mlp_d -> mlp_down

        # --- Norms ---
        b.attn_norm = _get(hf_block, cfg['norms']['attn'])  # ln1 -> attn_norm
        b.ffn_norm = _get(hf_block, cfg['norms']['ffn'])  # ln2 -> ffn_norm
        b.eps = getattr(_get(hf_block, cfg['norms']['attn'].rsplit('.', 1)[0]), 'eps', 1e-6)

        m.blocks.append(b)

    return m