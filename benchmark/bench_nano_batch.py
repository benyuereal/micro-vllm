"""nano-vllm 进程内批处理吞吐 bench，对齐 micro bench_batch_compare 的变长口径。

测什么：
  - 同 seed(0)，N 条 seqs，output 100 ~ MAX_OUT 随机，ignore_eos 跑满
  - 进程内 LLM.generate（无 HTTP），统计聚合吞吐 tok/s
  - 与 micro 的 HTTP 并发压测（bench_batch_compare.py）同 seed 同分布，口径对齐

用法：
  NUM_SEQS=256 MAX_OUT=1024 python3 bench_nano_batch.py
  环境变量 NUM_SEQS 控制请求数（默认 256），MAX_OUT 控制 max_tokens 上限（默认 1024）

依赖：
  - 空闲 GPU（CUDA_VISIBLE_DEVICES 由调用方指定，如 CUDA_VISIBLE_DEVICES=1）
  - nano-vllm 路径默认 /models/nano-vllm，可用环境变量 NANO_VLLM_PATH 覆盖
  - 模型路径默认 /models/Qwen3-0.6B，可用环境变量 MODEL_PATH 覆盖
"""
import os, sys, time
from random import randint, seed

NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

sys.path.insert(0, NANO_VLLM_PATH)
from nanovllm import LLM, SamplingParams

N = int(os.environ.get("NUM_SEQS", "256"))
MAX_OUT = int(os.environ.get("MAX_OUT", "1024"))

seed(0)
prompts = [f"Benchmark {i} " for i in range(N)]
max_tokens_list = [randint(100, MAX_OUT) for _ in range(N)]
total_out = sum(max_tokens_list)
sps = [SamplingParams(temperature=0.01, ignore_eos=True, max_tokens=m) for m in max_tokens_list]

llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)
# warmup
llm.generate(["warmup"], SamplingParams(temperature=0.01, max_tokens=4))

t0 = time.time()
llm.generate(prompts, sps, use_tqdm=False)
t = time.time() - t0
print(f"nano: N={N} total_out_tok={total_out} time={t:.2f}s throughput={total_out/t:.1f} tok/s")
