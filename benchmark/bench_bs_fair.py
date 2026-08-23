"""bs=32/64 单批次公平对比：micro-vllm vs nano-vllm。

测什么：
  - N 条同长 prompt、max_tokens 固定，全量 prefill+decode 计时
  - 输出聚合吞吐 tok/s（micro 按实际生成 token 计，nano 按 N*max_tokens 计，
    两者均 ignore_eos 跑满，口径一致）

用法：
  python3 bench_bs_fair.py <micro|nano> <N> [max_tokens]
  例: python3 bench_bs_fair.py micro 32 500
      python3 bench_bs_fair.py nano  64 500

依赖：
  - 空闲 GPU（CUDA_VISIBLE_DEVICES 由调用方指定，如 CUDA_VISIBLE_DEVICES=1）
  - micro 路径自动定位为本仓库根目录（脚本所在目录的上一级）
  - nano-vllm 路径默认 /models/nano-vllm，可用环境变量 NANO_VLLM_PATH 覆盖
  - 模型路径默认 /models/Qwen3-0.6B，可用环境变量 MODEL_PATH 覆盖

最新基准（L20 / Qwen3-0.6B / bf16，同 prompt，max_tokens=500）：
  bs=32: micro 7060 tok/s vs nano 6465 tok/s（+9.2%）
  bs=64: micro 9864 tok/s vs nano 9332 tok/s（+5.7%）
"""
import os, sys, time

# 自动定位仓库根目录（benchmark/ 的上一级），micro 代码从这里 import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

MODE = sys.argv[1]
N = int(sys.argv[2])
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 500
PROMPT = "The history of artificial intelligence began in the mid 20th century. " * 4

if MODE == "micro":
    sys.path.insert(0, REPO_ROOT)
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core import sequence as sm
    sm.Sequence.is_finished = lambda s: len(s.output_ids) >= s.max_tokens
    eng = InferenceEngine(MODEL, max_batch_size=max(N, 64), max_prefill_tokens=4096)
    for _ in range(N):
        eng.add_request(PROMPT, MAX_TOK, temperature=0.01, top_p=1.0)
    # warmup 一轮 prefill+少量 decode
    for _ in range(3):
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b); eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    torch_sync = __import__("torch").cuda.synchronize
    torch_sync(); t0 = time.time()
    n_tok = 0
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b); eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        n_tok += len(b)
    torch_sync(); dt = time.time() - t0
    print(f"micro bs={N}: {n_tok/dt:.0f} tok/s ({n_tok} tok, {dt:.3f}s)")
else:
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    llm.generate(["warmup"], SamplingParams(temperature=0.01, max_tokens=4))
    prompts = [PROMPT] * N
    sps = [SamplingParams(temperature=0.01, ignore_eos=True, max_tokens=MAX_TOK) for _ in range(N)]
    t0 = time.time()
    llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    print(f"nano  bs={N}: {N*MAX_TOK/dt:.0f} tok/s ({N*MAX_TOK} tok, {dt:.3f}s)")
