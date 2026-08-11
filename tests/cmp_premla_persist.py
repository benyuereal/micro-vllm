#!/usr/bin/env python3
"""pre-MLA persistent kernel token 一致性回归测试。

pre-MLA persistent 现为默认路径（无环境变量开关）。本脚本生成 N 个 token 并打印
id 序列，供与 vLLM/参考实现 diff 比对。序列应与未融合路径（kernel/pre_mla.py 的
3 个独立 kernel，见 tests/proto_premla_persist.py）完全一致。
"""
import sys, torch
sys.path.insert(0, "/models/micro-vllm")
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext

MODEL = "/models/DeepSeek-V2-Lite"
PROMPT = "请详细解释 Transformer 架构中多头自注意力机制的完整计算流程。"
N = 40


def main():
    engine = InferenceEngine(MODEL, max_batch_size=40)
    engine.add_request(PROMPT, max_tokens=N + 10, temperature=0.0)
    # prefill
    while True:
        b, bt = engine.get_next_batch()
        if bt == "waiting" or not b:
            continue
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        if bt == "prefill": break
    ids = []
    for _ in range(N):
        b, bt = engine.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        engine.step(ctx); engine.collect(ctx); engine.update_sequences(ctx.sequences)
        ids.append(ctx.sequences[0].output_ids[-1])
    print("IDS=" + ",".join(str(i) for i in ids))
    print("TEXT=" + engine.tokenizer.decode(ids))


if __name__ == "__main__":
    main()
