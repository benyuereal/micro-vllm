# Benchmark 脚本

所有吞吐/压测脚本集中在此目录。脚本自动定位仓库根目录（`REPO_ROOT`），
放哪都能跑；GPU 由调用方通过 `CUDA_VISIBLE_DEVICES` 指定（勿用 GPU0，
那是常驻 DeepSeek 服务）。

## 脚本一览

| 脚本 | 测什么 | 用法 |
|:-----|:-------|:-----|
| `fair_throughput.py` | **1000 请求连续批处理** micro vs nano 公平对比（同进程、同请求、同排空语义） | `python3 fair_throughput.py <N> <max_tokens> <micro\|nano\|both>` |
| `benchmark_throuput.py` | **单批次** bs=32/64 公平对比（同 prompt、同 max_tokens，全量 prefill+decode 计时） | `python3 benchmark_throuput.py <micro\|nano> <N> [max_tokens]` |
| `bench_batch_compare.py` | HTTP 并发 `/generate` 压测（N 条独立请求打 API 服务） | `python3 bench_batch_compare.py <url> <N>` |
| `bench_stream.py` | HTTP 并发流式 `/generate_stream` 压测（aiohttp） | `python3 bench_stream.py [batch_size]` |
| `bench_nano_batch.py` | nano-vllm 进程内批处理（变长 output，对齐 micro 口径） | `NUM_SEQS=256 MAX_OUT=1024 python3 bench_nano_batch.py` |

环境变量（可选）：`MODEL_PATH`（默认 `/models/Qwen3-0.6B`）、
`NANO_VLLM_PATH`（默认 `/models/nano-vllm`）、`API_URL`（默认 `http://localhost:8000`）。

## 最新基准数据（L20 / Qwen3-0.6B / bf16，2026-08-23）

### 1000 请求连续批处理（max_tokens 40-80 随机，temp=0.6，ignore_eos）

| 框架 | 吞吐 (tok/s) | 备注 |
|:-----|:-----------:|:-----|
| **micro-vllm** | **30,316** | 两轮 30316 / 30272，130 步 |
| nano-vllm | 27,638 | 27622 / 27653，153 步 |

micro 领先 **+9.7%**。micro 单步 GPU 时间（bs=512）13.5ms 低于 nano 14.36ms。

### 单批次吞吐（同 prompt，max_tokens=500，全量 prefill+decode 计时）

| batch | micro-vllm | nano-vllm | micro 优势 |
|:-----:|:----------:|:---------:|:---------:|
| 32 | **7,060** | 6,465 | +9.2% |
| 64 | **9,864** | 9,332 | +5.7% |

### 单用户 decode（短上下文 8 in / 200 out，temp=0.01，7 轮中位数）

| 框架 | 吞吐 (tok/s) |
|:-----|:-----------:|
| **micro-vllm** | **405.8** |
| vLLM 0.21.0 | 386.4 |
| nano-vllm | 335.8 |

## 压测指令

```bash
# 1000 请求连续批处理（micro / nano / both 同进程先后跑）
CUDA_VISIBLE_DEVICES=1 python3 fair_throughput.py 1000 80 micro
CUDA_VISIBLE_DEVICES=1 python3 fair_throughput.py 1000 80 nano
CUDA_VISIBLE_DEVICES=1 python3 fair_throughput.py 1000 80 both

# 单批次 bs=32/64
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throuput.py micro 32 500
CUDA_VISIBLE_DEVICES=2 python3 benchmark_throuput.py nano 32 500
CUDA_VISIBLE_DEVICES=1 python3 benchmark_throuput.py micro 64 500
CUDA_VISIBLE_DEVICES=2 python3 benchmark_throuput.py nano 64 500

# HTTP 并发压测（需先起服务: python3 api_server.py --model qwen3）
python3 bench_batch_compare.py http://localhost:8000 256
python3 bench_stream.py 32
```
