#!/usr/bin/env python3
"""vLLM 单用户 decode 吞吐基准（与 micro-vllm bench_e2e.py 同构对比）。
bs=1, 固定 prompt, CUDA event 计时稳态 decode（排除 prefill + 首轮热启动）。
直驱 vLLM LLM 引擎，不走 HTTP server。
GPU2（PID 431060 占 GPU0 跑 micro-vllm server，不能动）。
"""
import os, sys, time, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
from vllm import LLM, SamplingParams

MODEL = "/models/DeepSeek-V2-Lite"
PROMPT = "请详细解释 Transformer 架构中多头自注意力机制的完整计算流程，包括 Q K V 矩阵的生成、缩放点积注意力、softmax 归一化、多头拼接和输出投影。"
WARMUP_GEN = 60
MEASURE_GEN = 200

print("Loading vLLM engine ...", flush=True)
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    enforce_eager=False,          # 用 CUDA graph（与 micro-vllm 同条件）
    trust_remote_code=True,
    disable_log_stats=True,
)
sp = SamplingParams(temperature=0.0, max_tokens=WARMUP_GEN + MEASURE_GEN + 10)

# warmup（首轮含编译/图捕获开销，丢弃）
print("warmup ...", flush=True)
_ = llm.generate([PROMPT], sp)

# 正式测：bs=1 单请求，generate 返回后看 token 时间戳
# vLLM 的 RequestOutput 有 metrics? 用离线 generate + wall clock
# 更可靠：用 LLM.generate 单请求，timestamp 在 output
t0 = time.time()
outs = llm.generate([PROMPT], sp)
dt = time.time() - t0
out = outs[0]
n_gen = len(out.outputs[0].token_ids)
tps = n_gen / dt
print(f"\n=== vLLM bs=1 (wall clock, 含 prefill) ===")
print(f"gen tokens: {n_gen}, total {dt*1000:.0f} ms, {tps:.1f} tok/s")

# 稳态 decode：用 vLLM 的 arrival/finish time 精确量
# RequestOutput.metrics (vllm>=0.6): num_prompt_tokens, scheduler_time 等
# 更直接：单步驱动的 engine 不易暴露。改用：prefill 后单独量 decode 段
# 用两个 max_tokens 不同，差分出 decode 速率
sp_short = SamplingParams(temperature=0.0, max_tokens=WARMUP_GEN)
sp_long  = SamplingParams(temperature=0.0, max_tokens=WARMUP_GEN + MEASURE_GEN)
_ = llm.generate([PROMPT], sp_short)  # warmup
torch.cuda.synchronize()
t1 = time.time(); o1 = llm.generate([PROMPT], sp_short); dt1 = time.time() - t1
n1 = len(o1[0].outputs[0].token_ids)
torch.cuda.synchronize()
t2 = time.time(); o2 = llm.generate([PROMPT], sp_long);  dt2 = time.time() - t2
n2 = len(o2[0].outputs[0].token_ids)
# decode 段差分：(n2-n1) tok 用 (dt2-dt1) 秒（prefill 部分两次近似抵消）
decode_tps = (n2 - n1) / (dt2 - dt1)
print(f"\n=== vLLM bs=1 稳态 decode（差分法 n2-n1 / dt2-dt1）===")
print(f"short: {n1} tok {dt1*1000:.0f}ms | long: {n2} tok {dt2*1000:.0f}ms")
print(f"decode delta: {n2-n1} tok / {(dt2-dt1)*1000:.0f} ms = {decode_tps:.1f} tok/s")
print(f"\n=== micro-vllm bench_e2e: 112.7 tok/s ===")
print(f"=== vLLM (this): {decode_tps:.1f} tok/s ===")
print(f"=== micro-vllm / vLLM = {112.7/decode_tps:.2f}x ===")
