# Benchmark 脚本

所有吞吐/压测脚本集中在此目录。脚本自动定位仓库根目录（`REPO_ROOT`），
放哪都能跑；GPU 由调用方通过 `CUDA_VISIBLE_DEVICES` 指定（勿用 GPU0，
那是常驻 DeepSeek 服务）。

## 脚本一览（3 个）

| 脚本 | 测什么 | 用法 |
|:-----|:-------|:-----|
| `benchmark1000_throughput.py` | **1000 请求连续批处理** micro vs nano 公平对比（同进程、同请求、同排空语义） | `python3 benchmark1000_throughput.py <N> <max_tokens> <micro\|nano\|both>` |
| `benchmark_throuput.py` | **三方对比** micro vs vLLM 0.21.0 vs nano，128 in / 256 out，bs=1/32/64 聚合吞吐 | `python3 benchmark_throuput.py <micro\|vllm\|nano> <N>` |
| `benchmark_single_user.py` | **单用户长上下文** 256 in / 768 out（合计 1024），7 轮中位数，单请求 wall time | `python3 benchmark_single_user.py <micro\|vllm\|nano>` |

环境变量（可选）：`MODEL_PATH`（默认 `/models/Qwen3-0.6B`）、
`NANO_VLLM_PATH`（默认 `/models/nano-vllm`）。

## 最新基准数据（L20 / Qwen3-0.6B / bf16，2026-08-24）

### 三方对比 · 批次吞吐（128 in / 256 out，temp=0.01）

| 并发数 | micro-vllm | vLLM 0.21.0 | nano-vllm |
|:------:|:----------:|:-----------:|:---------:|
| 1      | **409.1**  | 386.2       | 340.9     |
| 32     | 7,503      | **7,749**   | 6,438     |
| 64     | 10,469     | **11,547**  | 9,635     |

bs=1 micro 领先 vLLM +6.0%；并发增大后 vLLM 凭 inductor 编译 + tensor-core GEMM 反超。

### 三方对比 · 单用户长上下文（256 in / 768 out，7 轮中位数）

| 框架 | 吞吐 (tok/s) |
|:-----|:-----------:|
| **micro-vllm** | **410.4** |
| vLLM 0.21.0 | 385.4 |
| nano-vllm | 347.1 |

micro 领先 vLLM +6.5%、nano +18.2%。长上下文下 micro 靠 bs=1 flash-decoding
（auto split-KV）把 KV 读并行到全部 SM，不掉速。

### 1000 请求连续批处理（max_tokens 40-80 随机，temp=0.6，ignore_eos）

| 框架 | 吞吐 (tok/s) |
|:-----|:-----------:|
| **micro-vllm** | **30,316** |
| nano-vllm | 27,638 |

micro 领先 +9.7%。

## 压测指令

```bash
# 1000 请求连续批处理（micro / nano / both 同进程先后跑）
CUDA_VISIBLE_DEVICES=1 python3 benchmark1000_throughput.py 1000 80 micro
CUDA_VISIBLE_DEVICES=1 python3 benchmark1000_throughput.py 1000 80 nano
CUDA_VISIBLE_DEVICES=1 python3 benchmark1000_throughput.py 1000 80 both

# 三方对比 bs=1/32/64（128 in / 256 out，temp=0.01，ignore_eos 跑满）
for f in micro vllm nano; do
  for bs in 1 32 64; do
    CUDA_VISIBLE_DEVICES=1 python3 benchmark_throuput.py $f $bs
  done
done

# 单用户长上下文（256 in / 768 out，7 轮中位数）
for f in micro vllm nano; do
  CUDA_VISIBLE_DEVICES=1 python3 benchmark_single_user.py $f
done
```
