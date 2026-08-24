"""公平连续批处理吞吐对比：micro-vllm vs nano-vllm。

测什么：
  - 同进程、同 N 个请求、同"先全部入队再排空"语义下的聚合 decode 吞吐
  - wall time 处理完全部请求、输出 token 总数、是否全部完成

micro: 一次性 add_request 全部请求，再跑 get_next_batch/step 循环到 idle
       （等同 nano 的 generate 语义）。
nano:  llm.generate(prompts, sampling_params)。
都用 Qwen3-0.6B，temp=0.6，ignore_eos 跑满 max_tokens 以测纯吞吐。

用法：
  python3 benchmark1000_throughput.py <N> <max_tokens> <micro|nano|both>
  例: python3 benchmark1000_throughput.py 1000 80 both

依赖：
  - 空闲 GPU（CUDA_VISIBLE_DEVICES 由调用方指定，如 CUDA_VISIBLE_DEVICES=1）
  - micro 路径自动定位为本仓库根目录（脚本所在目录的上一级）
  - nano-vllm 路径默认 /models/nano-vllm，可用环境变量 NANO_VLLM_PATH 覆盖
  - 模型路径默认 /models/Qwen3-0.6B，可用环境变量 MODEL_PATH 覆盖

最新基准（L20 / Qwen3-0.6B / bf16，N=1000，max_tokens 40-80 随机）：
  micro 28110 tok/s（三轮 28122/28098/28110） vs nano 27638 tok/s（27622/27653），micro +1.7%
"""
import os, sys, time, random

# 自动定位仓库根目录（benchmark/ 的上一级），micro 代码从这里 import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

NANO_VLLM_PATH = os.environ.get("NANO_VLLM_PATH", "/models/nano-vllm")
MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3-0.6B")

import torch
torch.manual_seed(0)

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 80
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"  # micro / nano / both

random.seed(0)
PROMPTS = [
    "写一个 SpringBoot 文件上传代码",
    "解释区块链的共识机制",
    "用JavaScript实现一个Promise限流池",
    "写一篇关于元宇宙未来的短文",
    "如何学习网络安全？给出学习路径",
    "比较SQL和NoSQL数据库的优缺点",
    "写一个关于时间旅行的科幻故事开头",
    "用Rust实现一个简单的链表结构",
    "写一篇关于远程工作利弊的分析",
    "如何成为一名全栈开发者？",
    "比较微服务和单体架构的优缺点",
    "解释机器学习中的过拟合与欠拟合",
    "用Python实现一个简单的神经网络",
    "如何系统地学习数据结构与算法？",
    "解释什么是碳中和以及实现路径",
]
prompts_list = [PROMPTS[i % len(PROMPTS)] for i in range(TOTAL)]
maxtoks_list = [random.randint(MAXTOK // 2, MAXTOK) for _ in range(TOTAL)]


def run_micro():
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=512, max_prefill_tokens=16384, max_context_length=1024)
    for i in range(TOTAL):
        eng.add_request(prompts_list[i], maxtoks_list[i], temperature=0.6, top_p=1.0)
    eng.scheduler.finished_sequences.clear()
    t0 = time.time()
    steps = 0
    while True:
        batch, batch_type = eng.get_next_batch()
        if batch_type == "idle" or not batch:
            if not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue:
                break
            continue
        ctx = BatchInferenceContext(len(batch), batch_type, batch)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        steps += 1
    wall = time.time() - t0
    out_tok = sum(len(s.output_ids) for s in eng.scheduler.finished_sequences)
    n_done = len(eng.scheduler.finished_sequences)
    return wall, out_tok, n_done, steps


def run_nano():
    sys.path.insert(0, NANO_VLLM_PATH)
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, enforce_eager=False, tensor_parallel_size=1, max_model_len=1024)
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt) for mt in maxtoks_list]
    llm.generate(["warmup"], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2), use_tqdm=False)
    t0 = time.time()
    outputs = llm.generate(prompts_list, sps, use_tqdm=False)
    wall = time.time() - t0
    out_tok = sum(len(o["token_ids"]) for o in outputs)
    return wall, out_tok, len(outputs)


if WHICH in ("micro", "both"):
    print(f"=== micro-vllm  TOTAL={TOTAL} maxtok~{MAXTOK} ===")
    w, ot, nd, st = run_micro()
    print(f"  完成: {nd}/{TOTAL}  wall={w:.2f}s  steps={st}")
    print(f"  输出 token 总数: {ot}")
    print(f"  聚合 decode 吞吐: {ot/w:.1f} tok/s")
    print(f"  请求速率: {TOTAL/w:.1f} req/s")
    print()
if WHICH in ("nano", "both"):
    print(f"=== nano-vllm   TOTAL={TOTAL} maxtok~{MAXTOK} ===")
    w, ot, nd = run_nano()
    print(f"  完成: {nd}/{TOTAL}  wall={w:.2f}s")
    print(f"  输出 token 总数: {ot}")
    print(f"  聚合 decode 吞吐: {ot/w:.1f} tok/s")
    print(f"  请求速率: {TOTAL/w:.1f} req/s")
