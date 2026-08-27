"""DFlash2 投机解码 engine 集成基准（Qwen3-0.6B 自起草）。

验证 + 性能：
1. 正确性：engine 的 SpecEngine（prepared 权重布局）投机解码输出
   与同权重 plain greedy（单 token/步，causal）逐 token 一致。
2. 性能：单用户 decode 吞吐对比（投机解码 vs plain greedy），平均接受 token 数。

用法：
  CUDA_VISIBLE_DEVICES=3 python3 benchmark/benchmark_spec_decode.py \
      --model /models/Qwen3-0.6B --N 3 --max-new 64
"""
import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plain_greedy_controller(ctrl, prompt_ids, max_new, eos=None):
    """用 controller 的 _forward 做 plain greedy（单 token/步，causal），作参考。
    与投机解码共用同一套 prepared 权重，隔离"投机机制"对输出的影响。"""
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
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is, and the reason is that")
    ap.add_argument("--mask-token", type=int, default=0)
    args = ap.parse_args()

    from core.engine import InferenceEngine
    engine = InferenceEngine(
        args.model, max_batch_size=8, max_context_length=2048,
        spec_decode=True, draft_model_path=None,
        num_speculative_tokens=args.N, mask_token_id=args.mask_token)
    ctrl = engine._spec_engine
    tokenizer = engine.tokenizer
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    print(f"投机解码: N={args.N} mask_token={args.mask_token} (自起草)")

    # 1. Plain greedy（参考，同 prepared 权重）
    t0 = time.time()
    ref = plain_greedy_controller(ctrl, prompt_ids, args.max_new, engine.eos_token_id)
    t_ref = time.time() - t0

    # 2. 投机解码
    t0 = time.time()
    spec = ctrl.generate(prompt_ids, args.max_new, eos_token_id=engine.eos_token_id)
    t_spec = time.time() - t0

    match = ref == spec
    print("\n" + "=" * 60)
    print(f"Plain greedy  : {len(ref)} tokens, {t_ref*1000:.1f}ms, {len(ref)/t_ref:.1f} tok/s")
    print(f"Spec decode   : {len(spec)} tokens, {t_spec*1000:.1f}ms, {len(spec)/t_spec:.1f} tok/s")
    print(f"Speedup       : {t_ref/t_spec:.2f}x")
    print(f"Output match  : {match}")
    if not match:
        for i, (a, b) in enumerate(zip(ref, spec)):
            if a != b:
                print(f"  首个分歧 @gen[{i}]: ref={a} spec={b}")
                print(f"  ref  : {tokenizer.decode(ref[:i+5])!r}")
                print(f"  spec : {tokenizer.decode(spec[:i+5])!r}")
                break
    print(f"平均接受 token 数: {ctrl.avg_acceptance:.3f} / N={args.N}")
    print("=" * 60)
    print(f"参考输出: {tokenizer.decode(ref)!r}")
    print(f"投机输出: {tokenizer.decode(spec)!r}")
    assert match, "engine 投机解码输出与 plain greedy 不一致！"
    print("\n✅ engine 集成验证通过：prepared 权重下投机解码与 plain greedy 逐 token 一致")


if __name__ == "__main__":
    main()
