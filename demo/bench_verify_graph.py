"""verify CUDA graph 前后 step time / tok/s 对比 + 3 次连续跑显存泄漏检查。

用法：CUDA_VISIBLE_DEVICES=4 MICRO_W8A16=1 python3 /vllm-workspace/tmp/bench_verify_graph.py
"""
import os, sys, time
REPO_ROOT = "/tmp/micro-vllm-verify-graph"
sys.path.insert(0, REPO_ROOT)
import torch
from core.engine import InferenceEngine

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.8-27B-INT8-W8A16-MTP")
DRAFT = os.environ.get("DRAFT_PATH", "/models/Qwen3.8-27B-DFlash2")
PROMPT = ("The history of artificial intelligence began in the mid 20th century. " * 4)
OUT_TOK = int(os.environ.get("OUT_TOK", "128"))
N_RUN = int(os.environ.get("N_RUN", "3"))


def mem_mb():
    return torch.cuda.memory_allocated() / (1 << 20)


def main():
    eng = InferenceEngine(MODEL, max_batch_size=16, max_prefill_tokens=4096,
                          spec_decode=True, draft_model_path=DRAFT,
                          num_speculative_tokens=7)
    ctrl = eng._spec_controller
    print(f"verify_graph={'ON' if ctrl._verify_graph is not None else 'OFF(eager)'}")

    # warmup（摊销 TileLang 编译 + graph 已捕获）
    _ = eng.generate_spec_decode(PROMPT, 32)
    torch.cuda.synchronize()

    base_mem = mem_mb()
    print(f"\nbaseline mem_allocated = {base_mem:.1f} MB")
    for i in range(N_RUN):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = eng.generate_spec_decode(PROMPT, OUT_TOK)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        tok = res["tokens"]
        steps = res["num_steps"]
        step_ms = wall / steps * 1000 if steps else 0
        cur = mem_mb()
        print(f"run{i+1}: {len(tok)} tok in {wall:.3f}s = {res['tok_s']:.1f} tok/s "
              f"steps={steps} step_time={step_ms:.2f}ms "
              f"accept={res['avg_acceptance']:.3f} "
              f"mem={cur:.1f}MB (Δ{cur-base_mem:+.1f})")
    print(f"\nfinal mem_allocated = {mem_mb():.1f} MB (Δ{mem_mb()-base_mem:+.1f})")


if __name__ == "__main__":
    main()
