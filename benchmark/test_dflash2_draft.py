"""DFlash2 草稿模型权重加载 + forward 冒烟测试（对齐 vLLM 机制）。

验证：
1. load_dflash2_draft 从 HF safetensors 加载真实权重，无未映射权重。
2. share_target_weights 共享 target 的 embed_tokens / lm_head。
3. forward（query=共享 embed，context_kv=aux hidden 经 fc+hidden_norm+k/v proj）跑通。
4. compute_candidates + select_draft_tokens（selector 贪心 walk）跑通。
5. 数值 sanity：输出非全 0 / 非 NaN。

用法：
  CUDA_VISIBLE_DEVICES=1 python3 benchmark/test_dflash2_draft.py
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "/models/Qwen3.8-27B-DFlash2"
N = 7  # num_speculative_tokens（block_size 8 = 1 bonus + 7 mask）
TARGET_HIDDEN = 5120  # Qwen3.8-27B hidden


def main():
    from models.dflash import load_dflash2_draft

    device = "cuda"
    dtype = torch.bfloat16
    print(f"加载 DFlash2 草稿模型: {MODEL} (N={N})")
    model, cfg = load_dflash2_draft(MODEL, dtype, device, N, max_pos=4096)
    model.eval()

    print(f"\n架构: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"q={cfg.num_attention_heads} kv={cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim} vocab={cfg.vocab_size}")
    print(f"dflash_config: {cfg.dflash_config}")
    print(f"mask_token_id={model.mask_token_id} target_layer_ids={model.target_layer_ids}")
    print(f"use_aux_hidden_state={model.use_aux_hidden_state} block_size={model.block_size}")
    print(f"candidate_selector={'有' if model.candidate_selector else '无'} "
          f"(top_k={getattr(model.candidate_selector, 'top_k', None)})")

    # 检查关键权重非随机（已加载真实值）
    w = model.layers[0].self_attn.q_proj.weight
    print(f"\nq_proj[0] weight: mean={w.float().mean().item():.5f} "
          f"std={w.float().std().item():.5f} (应非 0/非 1)")
    if model.candidate_selector is not None:
        cb = model.candidate_selector.predecessor_codebook
        print(f"predecessor_codebook: mean={cb.float().mean().item():.5f} "
              f"std={cb.float().std().item():.5f}")

    # ---- 共享 target 的 embed_tokens / lm_head（冒烟用随机占位，真实场景传 target 的）----
    target_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    target_embed.weight.data = target_embed.weight.data.to(dtype) * 0.02
    target_embed = target_embed.to(device).eval()
    target_lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    target_lm_head.weight.data = target_lm_head.weight.data.to(dtype) * 0.02
    target_lm_head = target_lm_head.to(device).eval()
    model.share_target_weights(target_embed, target_lm_head)
    print(f"\n已共享 target embed_tokens / lm_head")

    # ---- 构造 query（1 bonus + N mask）----
    T = 1 + N
    bonus = 100
    input_ids = torch.tensor([bonus] + [model.mask_token_id] * N, device=device)
    positions = torch.arange(100, 100 + T, device=device).long()
    input_embeds = model.embed_input_ids(input_ids)  # 共享 embed * scale
    print(f"query: input_ids={input_ids.tolist()}")
    print(f"input_embeds: {input_embeds.shape} (期望 [{T}, {cfg.hidden_size}])")

    # ---- 构造 context KV（target 中间层 hidden → fc → hidden_norm → k/v proj）----
    C = 32
    num_aux = len(model.target_layer_ids)
    # 模拟 target 中间层 hidden 拼接 [C, num_aux*target_hidden]
    aux = torch.randn(C, num_aux * TARGET_HIDDEN, device=device, dtype=dtype) * 0.02
    ctx_pos = torch.arange(100 - C, 100, device=device).long()
    context_states = model.combine_hidden_states(aux)  # fc → [C, hidden]
    context_kv = model.precompute_context_kv(context_states, ctx_pos)
    print(f"context_kv: {len(context_kv)} 层, 每层 k={context_kv[0][0].shape} "
          f"v={context_kv[0][1].shape}")

    # ---- forward（query + context_kv）----
    with torch.no_grad():
        h = model.forward(input_ids, positions, context_kv=context_kv,
                          input_embeds=input_embeds)
    print(f"\nforward 输出: {h.shape} (期望 [{T}, {cfg.hidden_size}])")
    assert h.shape == (T, cfg.hidden_size), f"形状错误: {h.shape}"
    assert torch.isfinite(h.float()).all(), "输出含 NaN/Inf"
    print(f"  mean={h.float().mean().item():.5f} std={h.float().std().item():.5f} "
          f"max={h.float().abs().max().item():.3f}")

    # ---- 无 context_kv 路径（纯 query）----
    with torch.no_grad():
        h2 = model.forward(input_ids, positions, input_embeds=input_embeds)
    assert h2.shape == (T, cfg.hidden_size)
    print(f"无 context_kv forward: {h2.shape} OK")

    # ---- compute_candidates + select_draft_tokens（mask 位置 hidden）----
    num_reqs = 2
    hs = torch.randn(num_reqs, N, cfg.hidden_size, device=device, dtype=dtype) * 0.02
    anchor_ids = torch.randint(0, cfg.vocab_size, (num_reqs,), device=device)
    with torch.no_grad():
        draft = model.select_draft_tokens(hs, anchor_ids)
    print(f"\nselect_draft_tokens: {draft.shape} (期望 [{num_reqs}, {N}])")
    assert draft.shape == (num_reqs, N)
    assert torch.isfinite(draft.float()).all()
    print(f"  draft tokens (req0): {draft[0].tolist()}")

    # ---- 真实 forward 的 mask 位置 hidden → select ----
    with torch.no_grad():
        real_h = model.forward(input_ids, positions, context_kv=context_kv,
                               input_embeds=input_embeds)
        mask_h = real_h[1:].unsqueeze(0)  # [1, N, hidden]（bonus 是第 0 个）
        anchor = torch.tensor([bonus], device=device)
        draft_real = model.select_draft_tokens(mask_h, anchor)
    print(f"真实 forward → select: {draft_real.shape} tokens={draft_real[0].tolist()}")

    print("\n" + "=" * 60)
    print("✅ DFlash2 草稿模型权重加载 + 共享权重 + forward + selector 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
