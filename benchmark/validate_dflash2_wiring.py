"""DFlash2 草稿路径 wiring 验证（合成草稿，随机权重）。

正确性不变量：投机解码输出 == target greedy 输出（与草稿质量无关，接受的 token
全是 target 自己的预测）。本脚本用真实 Qwen3-0.6B target + 合成 DFlash2 草稿
（随机权重，target_layer_ids 指向 target 中间层），验证 controller 的 DFlash2
交叉注意力草稿路径（_extract_aux → combine_hidden_states → precompute_context_kv
→ draft forward → target lm_head）wiring 正确：spec 输出与 plain greedy 逐 token 一致。

用法：
  CUDA_VISIBLE_DEVICES=3 python3 benchmark/validate_dflash2_wiring.py --max-new 32
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_dflash2(target_model, device, dtype, N, max_pos, num_aux=3):
    """构造合成 DFlash2 草稿（随机权重），target_layer_ids 指向 target 中间层。

    维度对齐 target（hidden/heads/kv/head_dim/intermediate/vocab），但只有 2 层
    sliding attn（DFlash2 结构：conv + selector + aux fc）。随机权重 → 草稿质量差，
    但 wiring 正确时 spec 输出仍 == target greedy。
    """
    from models.dflash.draft_model import DFlash2DraftModel

    cfg = target_model.config
    hidden = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    # target_layer_ids：均匀取 num_aux 个中间层
    step = max(1, num_layers // (num_aux + 1))
    target_layer_ids = [step * (i + 1) for i in range(num_aux)]
    target_layer_ids = [min(l, num_layers - 1) for l in target_layer_ids]

    class _Cfg:
        pass
    dcfg = _Cfg()
    dcfg.hidden_size = hidden
    dcfg.num_hidden_layers = 2  # 合成草稿只 2 层
    dcfg.num_attention_heads = cfg.num_attention_heads
    dcfg.num_key_value_heads = cfg.num_key_value_heads
    dcfg.head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
    dcfg.intermediate_size = cfg.intermediate_size
    dcfg.vocab_size = cfg.vocab_size
    dcfg.rms_norm_eps = cfg.rms_norm_eps
    dcfg.sliding_window = 2048
    dcfg.rope_parameters = {"rope_theta": getattr(cfg, "rope_theta", 1000000)}
    dcfg.dflash_config = {
        "block_size": 1 + N,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 0,
        "selector_rank": 32,
        "selector_top_k": 8,
        "target_layer_ids": target_layer_ids,
        "use_aux_hidden_state": True,
        "input_embedding_scale": 1.0,
    }
    dcfg.target_hidden_size = hidden
    draft = DFlash2DraftModel(dcfg, dtype, device, N, max_pos)
    # 随机权重（默认 init 已是随机，这里显式重设确保非零）
    with torch.no_grad():
        for p in draft.parameters():
            p.normal_(0, 0.02)
    draft.to(device)
    draft.eval()
    return draft, target_layer_ids


def plain_greedy_controller(ctrl, prompt_ids, max_new, eos=None):
    """plain greedy（单 token/步，causal），同 target 权重，作参考。"""
    ctrl.reset()
    positions = torch.arange(len(prompt_ids), device=ctrl.device).long()
    logits = ctrl._forward(ctrl.target, torch.tensor(prompt_ids, device=ctrl.device),
                           positions, 0, causal=True)
    tokens = list(prompt_ids)
    next_tok = int(logits[-1].argmax())
    tokens.append(next_tok)
    kv_len = len(prompt_ids)
    for _ in range(max_new - 1):
        if eos is not None and next_tok == eos:
            break
        pos = torch.tensor([kv_len], device=ctrl.device).long()
        logits = ctrl._forward(ctrl.target, torch.tensor([next_tok], device=ctrl.device),
                               pos, kv_len, causal=True)
        next_tok = int(logits[-1].argmax())
        tokens.append(next_tok)
        kv_len += 1
    return tokens[len(prompt_ids):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3-0.6B")
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prompt", default="The capital of France is, and the reason is that")
    ap.add_argument("--bench", action="store_true",
                    help="额外测 e2e 吞吐（spec decode vs plain greedy tok/s + 接受率）")
    args = ap.parse_args()

    from core.model_loader import load_model
    from core.spec_decode import SpecEngine

    device = "cuda:0"
    dtype = torch.bfloat16
    target_model, tokenizer = load_model(args.model, device=device)
    target_model.eval()

    draft, target_layer_ids = make_synthetic_dflash2(
        target_model, device, dtype, args.N, max_pos=2048)
    print(f"合成 DFlash2 草稿: 2 层, target_layer_ids={target_layer_ids}, "
          f"mask_token={draft.mask_token_id}")

    ctrl = SpecEngine(
        target_model, draft, device, dtype,
        num_speculative_tokens=args.N, mask_token_id=draft.mask_token_id,
        max_len=2048, draft_is_target=False)
    print(f"controller target_layer_ids={ctrl.target_layer_ids}")

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")

    ref = plain_greedy_controller(ctrl, prompt_ids, args.max_new)
    spec = ctrl.generate(prompt_ids, args.max_new)
    match = ref == spec
    print("\n" + "=" * 60)
    print(f"Plain greedy : {len(ref)} tokens")
    print(f"Spec (DFlash2): {len(spec)} tokens")
    print(f"Output match : {match}")
    if not match:
        for i, (a, b) in enumerate(zip(ref, spec)):
            if a != b:
                print(f"  首个分歧 @gen[{i}]: ref={a} spec={b}")
                break
    print(f"平均接受 token 数: {ctrl.avg_acceptance:.3f} / N={args.N} (合成草稿，预期低)")
    print("=" * 60)
    print(f"参考输出: {tokenizer.decode(ref)!r}")
    print(f"投机输出: {tokenizer.decode(spec)!r}")

    if args.bench:
        import time
        # e2e 吞吐：warmup 后计时 spec decode 与 plain greedy（合成草稿接受率≈0，
        # spec 每步 = 1 次 draft forward + 1 次 verify forward，故 tok/s 低于 greedy）。
        bench_new = max(args.max_new, 64)
        ctrl.generate(prompt_ids, 8)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec_b = ctrl.generate(prompt_ids, bench_new)
        torch.cuda.synchronize()
        t_spec = time.perf_counter() - t0
        ctrl.reset()
        t0 = time.perf_counter()
        ref_b = plain_greedy_controller(ctrl, prompt_ids, bench_new)
        torch.cuda.synchronize()
        t_greedy = time.perf_counter() - t0
        print("\n" + "=" * 60)
        print(f"e2e 吞吐（N={args.N}, max_new={bench_new}）:")
        print(f"  spec decode : {len(spec_b)} tok / {t_spec:.3f}s = {len(spec_b) / t_spec:.1f} tok/s  "
              f"接受率={ctrl.avg_acceptance:.3f}/N")
        print(f"  plain greedy: {len(ref_b)} tok / {t_greedy:.3f}s = {len(ref_b) / t_greedy:.1f} tok/s")
        print(f"  token match : {ref_b == spec_b}")
        print("=" * 60)

    assert match, "DFlash2 草稿路径 wiring 错误：spec 输出与 plain greedy 不一致！"
    print("\n✅ DFlash2 草稿路径 wiring 验证通过：交叉注意力草稿 + target lm_head，"
          "spec 输出与 plain greedy 逐 token 一致")


if __name__ == "__main__":
    main()
