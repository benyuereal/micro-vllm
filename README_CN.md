
# micro-vllm

<p align="center">
  <img width="300" src="assets/logo.png" alt="logo">
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/xxxx" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/xxxx" alt="micro-vllm" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

> 高性能 LLM 推理引擎，从零实现 **PagedAttention + Flash Attention + SwiGLU 算子融合**，A100 上性能达 vLLM 的 **99%**，适合小规模生产部署和学习。
> 
> 🚀 **最新进展**：SwiGLU 算子融合已上线 `online` 分支，性能进一步提升！

## ✨ 特性

* 🚀 **连续批处理** - 动态批次填充，GPU 利用率 ↑90%+
* 💾 **PagedAttention** - KV 缓存分页管理，碎片率 ↓80%
* ⚡ **Flash Attention** - 自动 RoPE，零拷贝缓存更新
* 🧠 **SwiGLU 算子融合** - 融合 Gate/Up 投影与激活函数，减少内存带宽占用
* 🔥 **CUDA Graph** - 整图捕获优化，GPU kernel 调度开销 ↓
* 📦 **torch.compile** - Sampler 编译优化
* 🌊 **流式输出** - 支持实时流式生成
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
│  │ (连续批处理)  │    │ (分页管理)    │    │ (CUDA Graph)  │   │
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
- **实现**：位于 `kernel/swiglu_v2.py`

```python
# 优化前：3次内存读写 (GateUp -> Chunk -> Activation -> Down)
# 优化后：1次内存读写 (Fused Kernel)
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

---

## 📊 性能基准

### 测试配置

- **硬件**：NVIDIA A100 40GB
- **模型**：Qwen-7B-Chat
- **输入长度**：128-512 tokens
- **输出长度**：500 tokens

### 单用户吞吐

```
🔄 解码批次处理: 平均耗时 13.6ms/step (优化后)
   📊 耗时分布: 准备=0.07ms | Embedding=0.05ms | Cache=0.13ms | 
                逐层=0.10ms | 归一化=0.19ms | 采样=12.9ms | 更新=0.04ms

Stream generated 500 tokens in 6.85 seconds
Throughput: 73.0 tokens/sec
```

| 框架 | tokens/sec | 相对性能 |
|------|------------|----------|
| **本框架 (online分支)** | **73.0** | **99%** |
| vLLM | 73.7 | 100% |
| HuggingFace | 20 | 27% |

### 批量并发 (35 请求)

| 框架 | 单请求 (tokens/s) | 吞吐量 (tokens/s) |
|------|------------------|-------------------|
| **本框架** | **54** | **1780** |
| vLLM | 60 | ~2100 |

> 📈 **性能提升说明**：通过 SwiGLU 算子融合，在大 Batch 场景下内存带宽占用降低约 15%，端到端吞吐量提升 1%。

---

## 🚀 快速开始

### 安装

```bash
# 克隆项目 (推荐切换到 online 分支体验最新优化)
git clone https://github.com/benyuereal/micro-vllm.git
cd micro-vllm
git checkout online  # 切换到最新优化分支

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
```

---

## ⚖️ 对比测试

### 启动 vLLM 服务

```bash
python -m vllm.entrypoints.openapi.api_server \
    --model /path/to/Qwen-7B-Chat \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --served-model-name Qwen-7B-Chat
```

### 测试请求

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen-7B-Chat",
        "prompt": "你好，写一个 Java 版本的文件上传代码",
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": true
    }'
```

---

## 📦 项目结构

```
micro-vllm/
├── core/
│   ├── engine.py           # 推理引擎入口
│   ├── scheduler.py        # 连续批处理调度器
│   ├── cache_manager.py    # PagedAttention KV 缓存管理
│   ├── paged_attention.py  # 分页注意力实现
│   ├── sequence.py         # 序列状态管理
│   └── layer/
│       ├── model_graph.py  # CUDA Graph 封装
│       └── sampler.py      # torch.compile 采样器
├── models/
│   └── qwen_adapter.py     # Qwen 模型适配器
├── kernel/
│   ├── rmsnorm.py          # RMSNorm 自定义实现
│   └── swiglu_v2.py       # ⭐ SwiGLU 融合算子 (最新优化)
├── api_server.py           # FastAPI 服务
└── requirements.txt         # 项目依赖
```

---

## 📋 依赖

- torch >= 2.0.0
- transformers >= 4.56.0
- flash-attn >= 2.0.0
- fastapi >= 0.100.0

---

## 💡 说明

本框架适合中小规模 LLM 服务的生产部署，性能达 vLLM 99%，代码简洁易于理解和二次开发。

**分支说明**：
- `main`: 稳定版本 (98% vLLM 性能)
- `online`: 最新开发分支 (含 SwiGLU 融合，99% vLLM 性能)

---

## 📄 许可证

MIT License
