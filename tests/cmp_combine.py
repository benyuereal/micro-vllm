"""对比新/旧 MLA combine kernel 的 e2e 输出是否 token-identical。
跑一次，存 200 token 到文件。配合 git stash 切换 kernel 版本跑两次。
用法：
  CUDA_VISIBLE_DEVICES=2 python3 tests/cmp_combine.py new   # 当前 kernel
  git stash                                              # 切旧 kernel
  CUDA_VISIBLE_DEVICES=2 python3 tests/cmp_combine.py old
  git stash pop
  python3 tests/cmp_combine.py diff
"""
import sys, torch, time, os, json
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

TAG = sys.argv[1] if len(sys.argv) > 1 else "new"
OUT = f"/tmp/combine_tokens_{TAG}.json"
PROMPT = "请详细解释 Transformer 架构中多头自注意力机制的完整计算流程，包括 Q K V 矩阵的生成、缩放点积注意力、softmax 归一化、多头拼接和输出投影。"
N = 200

if TAG == "diff":
    a = json.load(open("/tmp/combine_tokens_new.json"))
    b = json.load(open("/tmp/combine_tokens_old.json"))
    same = a == b
    print(f"new len={len(a)} old len={len(b)} identical={same}")
    if not same:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"first diff @ {i}: new={x} old={y}")
                print(f"new[:20]={a[:20]}")
                print(f"old[:20]={b[:20]}")
                break
    sys.exit(0)

engine = InferenceEngine("/models/DeepSeek-V2-Lite", max_batch_size=40)
engine.add_request(PROMPT, max_tokens=N + 10, temperature=0.0)
# prefill
while True:
    b, bt = engine.get_next_batch()
    if bt == "waiting" or not b:
        time.sleep(0.001); continue
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    if bt == "prefill": break
# decode N 步（bench_e2e 模式：不查 is_finished，靠 b 非空）
toks = []
for _ in range(N):
    b, bt = engine.get_next_batch()
    if not b: break
    ctx = BatchInferenceContext(len(b), bt, b)
    engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
    seq = ctx.sequences[0]
    if hasattr(seq, "output_ids"): toks = list(seq.output_ids)
tokens = toks[:N]
json.dump(tokens, open(OUT, "w"))
print(f"[{TAG}] saved {len(tokens)} tokens to {OUT}")
print(f"[{TAG}] first 20: {tokens[:20]}")
