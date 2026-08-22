# micro-vllm vs nano-vllm 单用户吞吐对比

> 测试日期：2026-08-22｜硬件：NVIDIA L20 (单卡)｜模型：Qwen3-0.6B bf16
> 口径：**进程内 Python API**（排除 HTTP 开销），单序列，`ignore_eos=True`，max_tokens=1024，temperature=0.01，warmup 1 次后取 3 次均值

## 结果

| 指标 | nano-vllm | micro-vllm | 差距 |
|---|---|---|---|
| 单用户 decode（engine 层，进程内） | **349.2 tok/s** | 288.3 tok/s | nano 快 21% |
| 单用户 decode（纯 GPU，排除 Python 开销） | — | 317.4 tok/s | nano kernel 快 ~10% |
| engine 层 Python 开销 | — | 0.38 ms/步（10%） | — |
| 多并发聚合吞吐（256 序列 / 8 并发 HTTP） | 5025 tok/s | ~311 tok/s | nano 快 16 倍 |

复现脚本：`/tmp/bench_micro_single.py`（micro，GPU1）、`/tmp/bench_nano_single.py`（nano，GPU3，需 `PYTHONPATH=/models/nano-vllm`）。
micro 纯 GPU 拆分脚本：`/tmp/bench_micro_single2.py`（用 cuda.Event 量 graph_runner.decode 的纯 GPU 时间，排除 collect/update_sequences/sampler 的 Python 开销）。

## 为什么 micro-vllm 单用户反而更慢

> 旧记录称「micro 单用户 428 tok/s，比 nano 350 快 22%」——**该数据无法复现**，疑为 DeepSeek 路径或误记。
> 公平进程内对比下，micro 单用户 decode **慢于** nano。根因分两层：

### 1. decode kernel 本身慢约 10%（317 vs 349）

即使排除所有 Python 开销，micro 纯 GPU decode 速率（317 tok/s）仍低于 nano（349 tok/s）。两者都用 CUDA Graph replay，差异在 attention/KV 写回实现：

- **nano**：用 Triton `store_kvcache` kernel（`slot_mapping` 索引）把 K/V 写进 paged cache，再调 `flash_attn_with_kvcache`。KV 写回与 attention 解耦，slot_mapping 是紧凑连续索引。
- **micro**：直接用 `flash_attn_with_kvcache(block_table=...)` 的 block_table 间接寻址路径，KV 写回与 attention 在同一个 flash_attn 调用内，block_table 二级间接寻址。

block_table 二级间接寻址相比 slot_mapping 紧凑索引，在单序列短上下文下可能有额外的地址计算/访存模式开销。**这是 attention 实现差异，有 ~10% 优化空间**——值得对比 nano 的 Triton store_kvcache 实现。

### 2. engine 层 Python 开销再吃 10%（317 → 288）

micro 的 `engine.generate()` 每步 decode 执行 `step → collect → update_sequences`，含 sampler、sequence 状态机更新、stop 串检查等 Python 逻辑，实测 0.38 ms/步。1024 步累计 ~386 ms，把纯 GPU 的 317 拖到 engine 层 288（−10%）。

nano 的 engine 层开销更小（349 接近其 kernel 上限），调度路径更精简。

## 多并发差 16 倍：100% 在调度

这是更大的问题。micro 的 `/generate` 端点调用 `engine.generate()`——一个**自包含的同步事件循环**，跑完当前请求的所有 token 才返回。8 个并发 HTTP 请求各自独立调 `engine.generate()`，被串行化处理（8 × 682 ms = 5.46 s），**没有跨请求的连续批处理（continuous batching）**。

nano 用 continuous batching + chunked prefill，256 序列同时 decode，5025 tok/s。

api_server.py 其实已有后台 `rank0_inference_loop` 和 `/generate_stream`（用 `add_request` + 回调），但非流式 `/generate` 绕过了它，走了串行 `engine.generate()` 路径。

## 结论与优先级

| 优化项 | 预期收益 | 优先级 |
|---|---|---|
| `/generate` 端点改走调度器连续批处理（修 16 倍多并发差距） | 多并发 16× | **最高** |
| 对比 nano Triton store_kvcache，优化 micro decode attention | 单用户 ~10% | 高 |
| 连续内存 gather（用户最初设想） | 低（paged 不是瓶颈） | 低 |

- 用户最初设想的「连续内存对 flash-attn 友好」——实测 paged block_table 不是 16 倍差距的瓶颈，单用户差距也不在此。优先级降低。
- **调度重构（连续批处理真正生效）是 16 倍差距的唯一解**，也是用户任务 3「参考 vllm model runner v2 固定停车位连续内存装载」的核心目标。
