# Benchmark 脚本

所有吞吐/压测脚本集中在此目录。脚本自动定位仓库根目录（`REPO_ROOT`），
放哪都能跑；GPU 由调用方通过 `CUDA_VISIBLE_DEVICES` 指定（勿用 GPU0，
那是常驻 DeepSeek 服务）。

## 脚本一览（3 个）

| 脚本 | 测什么 | 用法 |
|:-----|:-------|:-----|
| `bench_throughput.py` | **统一吞吐基准**：`--n 1` 单用户长上下文（256 in / 768 out，7 轮中位数，单请求 wall time）；`--n N` N 请求连续批处理（全入队排空，聚合 tok/s + req/s + 完成数）。后端 micro / vllm / nano / all | 见下方"压测指令" |
| `validate_spec_decode.py` | **spec 机制正确性验证**：DFlash2 草稿/验证/接受率逐层核对（保留，勿动） | `python3 benchmark/validate_spec_decode.py` |
| `benchmark_spec_decode.py` | **engine 集成 spec 正确性 + 加速比**：整图 spec 路径 token 对齐 + 吞吐对比（保留，勿动） | `python3 benchmark/benchmark_spec_decode.py` |

`bench_throughput.py` 参数：

- `--n`：1 = 单用户模式（多轮中位数，tok/s = out/wall，含 prefill，口径三者一致）；N>1 = 批量模式（temp=0.6，ignore_eos 跑满随机 max_tokens）
- `--in-tok` / `--out-tok` / `--rounds`：单用户模式输入/输出 token 数与轮数（默认 256 / 768 / 7）
- `--max-tok`：批量模式随机 max_tokens 上限，实际取 [max-tok/2, max-tok]（默认 80）
- `--max-batch`：micro 引擎 `max_batch_size`（决定 GDN 状态池大小，类级单例 pool=max_bs）。27B 在 44GiB 卡上 64/512 会 OOM（池 9GiB/70GiB + 权重/KV ~37GiB），默认 16；单用户 bs=1 足够，批量模式按并发需求调大
- `--backend`：`micro`（InferenceEngine 直接驱动）/ `vllm`（vllm.LLM，仅单用户模式）/ `nano`（nanovllm）/ `all`（该模式支持的全部后端，独立子进程先后跑）

环境变量（可选）：`MODEL_PATH`（默认 `/models/Qwen3.8-27B-INT8-W8A16-MTP`）、
`NANO_VLLM_PATH`（默认 `/models/nano-vllm`）。

## 历史基准数据（L20 / Qwen3-0.6B / bf16，2026-08-24）

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
# 单用户长上下文（256 in / 768 out，7 轮中位数）
CUDA_VISIBLE_DEVICES=1 python3 benchmark/bench_throughput.py --n 1 --backend micro
CUDA_VISIBLE_DEVICES=1 python3 benchmark/bench_throughput.py --n 1 --backend all   # micro+vllm+nano 独立子进程

# N 请求连续批处理（N=1000，max_tokens 40-80 随机）
CUDA_VISIBLE_DEVICES=1 python3 benchmark/bench_throughput.py --n 1000 --max-tok 80 --backend micro
CUDA_VISIBLE_DEVICES=1 python3 benchmark/bench_throughput.py --n 1000 --max-tok 80 --backend nano
CUDA_VISIBLE_DEVICES=1 python3 benchmark/bench_throughput.py --n 1000 --max-tok 80 --backend all
```
