
# micro-vllm

<p align="center">
  <img width="300" src="assets/logo.png" alt="logo">
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/xxxx" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/xxxx" alt="micro-vllm" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

> 高性能 LLM 推理引擎，从零实现 **PagedAttention + Flash Attention + 手写 CUDA GEMV + SwiGLU 算子融合**，L20 上单用户长上下文吞吐达 vLLM 的 **106%**、nano-vllm 的 **118%**，适合小规模生产部署和学习。
> 
> 🚀 **最新进展**：单用户长上下文（256 in / 768 out）达 **410.4 tok/s**，领先 vLLM **+6.5%**——靠 bs=1 的 flash-decoding（auto split-KV）+ 一个 paged-KV off-by-one 修复。1000 请求连续批处理保持 **30,316 tok/s**（领先 nano-vllm +9.7%）。

## ✨ 特性

* 🚀 **连续批处理** - 动态批次填充，batch=32 时 GPU 利用率 **~99%**
* 💾 **PagedAttention** - KV 缓存分页管理，碎片率 ↓80%
* ⚡ **Flash Attention** - 自动 RoPE，零拷贝缓存更新
* 🧠 **SwiGLU 算子融合** - 融合 Gate/Up 投影与激活函数，减少内存带宽占用
* 🔥 **CUDA Graph** - 整图捕获优化，GPU kernel 调度开销 ↓
* 📦 **torch.compile** - Sampler 编译优化
* 🌊 **流式输出** - 支持实时流式生成
* 🌐 **张量并行 (Tensor Parallelism)** - 支持多卡并行推理，突破单卡显存限制
* 📖 **简洁代码** - 约 1500 行 Python 代码，易于学习和二次开发

---

## 📚 目录

- [特性](#-特性)
- [架构设计](#-架构设计)
- [核心技术](#-核心技术)
- [性能基准](#-性能基准)
- [快速开始](#-快速开始)
- [API 参考](#-api-参考)
- [对比测试](#-对比测试)

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        InferenceEngine                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Scheduler  │───▶│  KVCacheMgr  │───▶│ModelGraphRunner│   │
│  │ (连续批处理)  │    │ (分页管理)    │    │(TP+CUDA Graph)│   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Flash Attention v2                    │  │
│  │              flash_attn_with_kvcache                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 SwiGLU Fused Kernel                      │  │
│  │              (Gate + Up + Activation)                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 职责 |
|------|------|
| `InferenceEngine` | 推理引擎入口，自动模型加载和配置 |
| `Scheduler` | 连续批处理调度，SJF 对齐策略 |
| `KVCacheManager` | PagedAttention KV 缓存分页管理 |
| `ModelGraphRunner` | CUDA Graph 整图捕获和执行 |
| `Sampler` | torch.compile 编译的 Token 采样器 |

---

## 🔬 核心技术

### 1. PagedAttention

参考 [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) 实现：

- **机制**：KV 缓存分页（Block=256 tokens），动态分配和释放
- **优势**：碎片率 5%，复用率 92%，避免预分配内存浪费

```python
# 核心接口
cache_manager.alloc(seq_id, num_tokens)  # 分配缓存块
cache_manager.append(seq_id)             # 追加新 token
cache_manager.free(seq_id)               # 释放缓存
```

### 2. Flash Attention v2

使用 `flash_attn_with_kvcache` 实现高效注意力：

- **自动 RoPE**：传入 `rotary_cos/sin` 即可
- **零拷贝**：直接更新到已有 KV 缓存
- **Paged KV**：支持 `block_table` 分页访问

```python
flash_attn_with_kvcache(
    q=q.unsqueeze(1),
    k_cache=k_cache,
    v_cache=v_cache,
    rotary_cos=cos_cache,
    rotary_sin=sin_cache,
    block_table=block_table,
    causal=True
)
```

### 3. SwiGLU 算子融合 (NEW ⭐)

使用自定义 Kernel 融合 MLP 层的计算瓶颈：

- **机制**：将 Gate Proj、Up Proj 矩阵乘法与 SwiGLU 激活融合为单个 Kernel
- **优势**：减少中间结果的 HBM 读写，显著降低内存带宽压力，特别提升大 Batch 场景吞吐量
- **实现**：位于 `kernel/swiglu.py`

```python
from kernel.swiglu import swiglu_fused
activated = swiglu_fused(gate_up)  # 一步完成融合计算
```

### 4. CUDA Graph 整图优化

将所有 Transformer 层封装到单个 CUDA Graph：

- **机制**：捕获 N 层前向为一个 Graph，一次 replay 完成
- **优势**：消除层间调度 overhead，支持多 batch_size 预捕获
- **支持**：batch_size ∈ [1, 2, 4, 8, 16, 32]

### 5. torch.compile 采样优化

使用 PyTorch compile 编译整个采样过程：

- **融合内核**：Top-K + Top-P 过滤在一个 kernel 内完成
- **动态 batch**：支持不同 batch_size
- **模式**：`reduce-overhead` 减少 Python 开销

### 6. 连续批处理调度

解码阶段采用连续批处理策略：

| 策略 | 实现 | 目标 |
|------|------|------|
| **动态填充** | 新请求随时插入 prefill | 最大化 GPU 利用率 |
| **同长度成批** | 相同长度序列组成批次 | 消除 padding 浪费 |
| **SJF 对齐** | 短序列优先完成 | 形成"长度簇" |

> **典型对齐过程**：
> ```
> t=0: [50, 52, 55, 60, 100] → 选长度 50
> t=1: [51, 52, 55, 60, 100] → 选长度 51
> t=2: [52, 52, 55, 60, 100] → 选长度 52（两序列对齐）
> ...
> t=8: [60, 60, 60, 60, 100] → 四序列完美对齐！
> ```

### 7. 张量并行 (Tensor Parallelism) (NEW ⭐)

支持多 GPU 分布式推理，突破单卡显存限制：

- **切分策略**：Column Parallel + Row Parallel
  - MLP: Gate/Up 投影Column Parallel，Down 投影 Row Parallel
  - Attention: QKV 投影 Column Parallel，Output 投影 Row Parallel
- **通信**：使用 `all_reduce` 汇总激活，确保语义一致
- **优势**：支持超大模型部署，保持高效推理

---

## 📊 性能基准

### 三方对比 · 单用户吞吐（L20 / Qwen3-0.6B）

与 vLLM、nano-vllm 在完全对齐条件下公平对比。长上下文（256 in / 768 out）压测 decode 期间的 KV cache 读取——attention 实现与 flash-decoding 并行度差异最明显的区间：

> **硬件**：NVIDIA L20 &nbsp;|&nbsp; **模型**：Qwen3-0.6B (bf16) &nbsp;|&nbsp; **输入**：256 tokens &nbsp;|&nbsp; **输出**：768 tokens &nbsp;|&nbsp; **采样**：temperature=0.01 &nbsp;|&nbsp; **方法**：7 轮取中位数，各引擎独占一张 GPU

| 框架 | 吞吐 (tokens/s) | 相对性能 |
|:-----|:----------------:|:--------:|
| **micro-vllm** | **410.4** | **1.18×** |
| vLLM 0.21.0 | 385.4 | 1.11× |
| nano-vllm | 347.1 | 1.00× |

- micro-vllm 单用户长上下文吞吐领先 vLLM **+6.5%**、领先 nano-vllm **+18.2%**
- 三引擎 7 轮测量标准差均 < 1 token/s，性能稳定可复现
- 优势来源：手写 CUDA GEMV + CUDA Graph 在 M=1 decode 下摊薄 kernel 固定开销，加上 flash-decoding（auto split-KV）让 KV 读取随上下文增长仍并行到全部 SM

### 三方对比 · 批次吞吐（L20 / Qwen3-0.6B）

并发吞吐随 batch 变化的趋势（单用户是 micro-vllm 优势区，batch 增大后 vLLM 的 inductor 编译 + tensor core GEMM 反超）：

> **硬件**：NVIDIA L20 &nbsp;|&nbsp; **模型**：Qwen3-0.6B (bf16) &nbsp;|&nbsp; **输入**：128 tokens &nbsp;|&nbsp; **输出**：256 tokens &nbsp;|&nbsp; **采样**：temperature=0.01 &nbsp;|&nbsp; **各引擎独占一张 GPU**

| 并发数 | micro-vllm | vLLM 0.21.0 | nano-vllm |
|:------:|:----------:|:-----------:|:---------:|
| 1      | **409.1**  | 386.2       | 340.9     |
| 32     | 7,503      | **7,749**   | 6,438     |
| 64     | 10,469     | **11,547**  | 9,635     |

- 单用户（bs=1）micro-vllm 领先 vLLM +6.0%；并发增大后 vLLM 凭借编译优化与 tensor core GEMM 反超
- 三方均支持连续批处理，batch=64 时系统总吞吐均破万 tok/s
- micro-vllm 定位清晰：**低并发延迟敏感场景**（手写 GEMV + 整图 Graph 摊薄 kernel 固定开销），而非高并发吞吐场景

### 连续批处理 · 1000 请求（L20 / Qwen3-0.6B）

1000 条混合请求（max_tokens 40-80 随机，temp=0.6，ignore_eos），全部入队后排空——真实高并发服务场景：

> **硬件**：NVIDIA L20 &nbsp;|&nbsp; **模型**：Qwen3-0.6B (bf16) &nbsp;|&nbsp; **方法**：三轮稳定值

| 框架 | 吞吐 (tok/s) | 步数 |
|:-----|:-----------:|:----:|
| **micro-vllm** | **30,316** | 130 |
| nano-vllm | 27,638 | 153 |

- micro-vllm 领先 nano-vllm **+9.7%**；bs=512 单步 GPU 时间也更低（13.5ms vs 14.36ms）
- 近期优化：bs=1 decode 的 flash-decoding（auto split-KV，长上下文 361→410 tok/s，+13.6%——16 CTA→全部 92 SM）、paged-KV off-by-one 修复（prefill 长度恰为 block_size 整数倍时首 decode 步越界崩溃）、Gumbel-max 单 Triton 采样 kernel（1225→269us/步，免 311MB fp32 物化）、`update_sequences` decode 稳态快速路径（省 0.75ms/步 CPU）、采样器去 `reduce-overhead`（省 155MB logits DtoD 拷贝，410us/步）、QK-Norm+RoPE 单 kernel 融合、`prepare()` 脏标志（稳态 CPU 0.88ms→0）、final-norm 融合
- 单批次（同 prompt，500 out，全量 prefill+decode）：bs=32 **7,060** vs 6,465（+9.2%）、bs=64 **9,864** vs 9,332（+5.7%）

压测脚本与压测指令见 [`benchmark/`](benchmark/README.md)。



---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/benyuereal/micro-vllm.git
cd micro-vllm
# 安装依赖
pip install -r requirements.txt
```

### 模型下载

```bash
huggingface-cli download --resume-download Qwen/Qwen2-7B-Chat \
  --local-dir ~/huggingface/Qwen2-7B-Chat/ \
  --local-dir-use-symlinks False
```

### 基础用法

```python
from core.engine import InferenceEngine

# 初始化引擎
engine = InferenceEngine(
    model_path="/path/to/Qwen2-7B-Chat",
    max_batch_size=32
)

# 批量生成
results = engine.generate(
    ["Hello", "AI is"],
    max_tokens=100
)
for prompt, text in results.items():
    print(f"{prompt}: {text}")

# 流式生成
for token, text in engine.stream_generate("AI 的未来是", max_tokens=50):
    print(text, end="", flush=True)
```

### 启动 API 服务

```bash
python api_server.py
```

服务启动后可访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 🌐 API 参考

### 非流式生成

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一个 Java 版本的文件上传代码",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### 流式生成

```bash
curl -X POST "http://localhost:8000/generate_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一个 SpringBoot 文件上传代码",
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": true
  }'
  
  curl -X POST "http://localhost:8000/generate_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一个 SpringBoot + vue 文件上传代码的完整解决方案",
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": true
  }'
  
curl -X POST "http://localhost:8000/generate_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "介绍一下北京这个城市,还有沈阳 天津",
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": true
  }'
```

---

## ⚖️ 对比测试

### 启动 vLLM 服务

```bash
vllm serve /models/Qwen3-0.6B --port 8001 --max-model-len 4096
```


---



---

## 📋 依赖

- torch >= 2.0.0
- transformers >= 4.56.0
- flash-attn >= 2.0.0
- fastapi >= 0.100.0

---

## 💡 说明

本框架适合中小规模 LLM 服务的生产部署，单用户吞吐达 vLLM 105%，代码简洁易于理解和二次开发。

---

## 📄 许可证

MIT License
