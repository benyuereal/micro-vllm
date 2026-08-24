"""三方对比：micro-vllm vs vLLM 0.21.0 vs nano-vllm（同进程内、同配置、同 prompt）。

测什么：
  - N 条 128-token prompt、max_tokens=256（ignore_eos 跑满）、temp=0.01
  - 聚合 decode 吞吐 tok/s = N*256 / wall time
  - 每个框架独立进程跑（避免显存/编译互相干扰），GPU 由调用方指定

用法：
  python3 benchmark_three_way.py <micro|vllm|nano> <N>
  例: python3 benchmark_three_way.py micro 32
      python3 benchmark_three_way.py vllm  32
      python3 benchmark_three_way.py nano  32

依赖：
  - 空闲 GPU（CUDA_VISIBLE_DEVICES 由调用方指定，如 CUDA_VISIBLE_DEVICES=1）
  - micro 路径自动定位为本仓库根目录（脚本所在目录的上一级）
  - nano-vllm 路径默认 /models/nano-vllm，可用环境变量 NANO_VLLM_PATH 覆盖
  - 模型路径默认 /models/Qwen3-0.6B，可用环境变量 MODEL_PATH 覆盖
"""
import os, sys, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

MODE = sys.argv[1]
N = int(sys.argv[2])
IN_TOK = 128    # 输入固定 128 token
OUT_TOK = 256   # 输出固定 256 token（ignore_eos 跑满）
TEMP = 0.01


def make_prompt():
    """构造恰好 128 token 的 prompt（用模型自带 tokenizer 对齐）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The history of artificial intelligence began in the mid 20th century. "
    ids = tok.encode(base)
    while len(ids) < IN_TOK:
        ids += tok.encode(" data")
    ids = ids[:IN_TOK]
    return tok.decode(ids)


if MODE == "micro":
    sys.path.insert(0, REPO_ROOT)
    import torch
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from core import sequence as sm
    sm.Sequence.is_finished = lambda s: len(s.output_ids) >= s.max_tokens
    eng = InferenceEngine(MODEL, max_batch_size=max(N, 64), max_prefill_tokens=4096)
    prompt = make_prompt()
    # 独立 warmup（1 条短请求跑满，触发 graph 捕获/编译）
    eng.add_request(prompt, 8, temperature=TEMP, top_p=1.0)
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
    for _ in range(N):
        eng.add_request(prompt, OUT_TOK, temperature=TEMP, top_p=1.0)
    torch.cuda.synchronize(); t0 = time.time()
    n_tok = 0
    while True:
        b, bt = eng.get_next_batch()
        if not b: break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        n_tok += len(b)
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"micro bs={N}: {n_tok/dt:.1f} tok/s ({n_tok} tok, {dt:.3f}s)")

elif MODE == "vllm":
    from vllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt()
    llm.generate([prompt], SamplingParams(temperature=TEMP, max_tokens=8))
    prompts = [prompt] * N
    sps = [SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=OUT_TOK) for _ in range(N)]
    t0 = time.time()
    llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    print(f"vllm  bs={N}: {N*OUT_TOK/dt:.1f} tok/s ({N*OUT_TOK} tok, {dt:.3f}s)")

elif MODE == "nano":
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
    prompt = make_prompt()
    llm.generate([prompt], SamplingParams(temperature=TEMP, max_tokens=8))
    prompts = [prompt] * N
    sps = [SamplingParams(temperature=TEMP, ignore_eos=True, max_tokens=OUT_TOK) for _ in range(N)]
    t0 = time.time()
    llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    print(f"nano  bs={N}: {N*OUT_TOK/dt:.1f} tok/s ({N*OUT_TOK} tok, {dt:.3f}s)")

else:
    raise SystemExit(f"unknown mode: {MODE} (micro|vllm|nano)")
