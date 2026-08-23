
# micro-vllm

<p align="center">
  <img width="300" src="assets/logo.png" alt="logo">
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/xxxx" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/xxxx" alt="micro-vllm" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

> A high-performance LLM inference engine implementing **PagedAttention + Flash Attention + Hand-written CUDA GEMV + SwiGLU Kernel Fusion** from scratch. Achieves **105%** of vLLM and **121%** of nano-vllm single-user throughput on L20, suitable for small-scale production deployment and learning.
> 
> 🚀 **Latest Update**: Hand-written CUDA GEMV integrated into the decode hot path, single-user throughput up another **+3.8%**!

## ✨ Features

* 🚀 **Continuous Batching** - Dynamic batch filling, GPU utilization **~99%** at batch=32
* 💾 **PagedAttention** - KV cache paging management, fragmentation ↓80%
* ⚡ **Flash Attention** - Automatic RoPE, zero-copy cache update
* 🧠 **SwiGLU Kernel Fusion** - Fused Gate/Up projection with activation to reduce memory bandwidth usage
* 🔥 **CUDA Graph** - Whole-graph capture optimization, GPU kernel scheduling overhead ↓
* 📦 **torch.compile** - Sampler compilation optimization
* 🌊 **Streaming Output** - Real-time streaming generation support
* 🌐 **Tensor Parallelism** - Multi-GPU distributed inference, break single-GPU memory limits
* 📖 **Clean Codebase** - ~1500 lines of Python code, easy to learn and extend

---

## 📚 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Core Technologies](#-core-technologies)
- [Performance Benchmark](#-performance-benchmark)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Comparison](#-comparison)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        InferenceEngine                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Scheduler  │───▶│  KVCacheMgr  │───▶│ModelGraphRunner│   │
│  │(Continuous  │    │   (Paging)   │    │(TP+CUDA Graph)│   │
│  │  Batching)  │    │              │    │              │   │
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

### Core Components

| Component | Responsibility |
|-----------|----------------|
| `InferenceEngine` | Inference engine entry, auto model loading |
| `Scheduler` | Continuous batching, SJF alignment strategy |
| `KVCacheManager` | PagedAttention KV cache paging |
| `ModelGraphRunner` | CUDA Graph capture and execution |
| `Sampler` | torch.compile compiled token sampler |

---

## 🔬 Core Technologies

### 1. PagedAttention

Implemented based on [vLLM PagedAttention](https://arxiv.org/abs/2309.06180):

- **Mechanism**: KV cache paging (Block=256 tokens), dynamic allocation
- **Benefits**: 5% fragmentation, 92% reuse rate, no pre-allocation waste

```python
# Core API
cache_manager.alloc(seq_id, num_tokens)  # Allocate cache blocks
cache_manager.append(seq_id)             # Append new token
cache_manager.free(seq_id)               # Free cache
```

### 2. Flash Attention v2

Using `flash_attn_with_kvcache` for efficient attention:

- **Auto RoPE**: Pass `rotary_cos/sin` directly
- **Zero-copy**: Update directly to existing KV cache
- **Paged KV**: Support `block_table` paging access

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

### 3. SwiGLU Kernel Fusion (NEW ⭐)

Custom kernel to fuse the MLP layer bottleneck:

- **Mechanism**: Fuses Gate Projection, Up Projection matrix multiplication, and SwiGLU activation into a single kernel
- **Benefits**: Reduces intermediate HBM reads/writes, significantly lowers memory bandwidth pressure, especially improving throughput in large batch scenarios
- **Implementation**: Located in `kernel/swiglu.py`

```python
from kernel.swiglu import swiglu_fused
activated = swiglu_fused(gate_up)  # Fused computation in one step
```

### 4. CUDA Graph Optimization

Encapsulate all Transformer layers into a single CUDA Graph:

- **Mechanism**: Capture N-layer forward as one Graph, single replay
- **Benefits**: Eliminate inter-layer scheduling overhead, pre-capture multiple batch sizes
- **Supported**: batch_size ∈ [1, 2, 4, 8, 16, 32]

### 5. torch.compile Sampling Optimization

Compile the entire sampling process using PyTorch compile:

- **Fused Kernel**: Top-K + Top-P filtering in one kernel
- **Dynamic Batch**: Support different batch sizes
- **Mode**: `reduce-overhead` to reduce Python overhead

### 6. Continuous Batching Scheduler

Continuous batching strategy in decode phase:

| Strategy | Implementation | Goal |
|----------|---------------|------|
| **Dynamic Fill** | New requests insert prefill anytime | Maximize GPU utilization |
| **Same Length Batch** | Sequences with same length form batch | Eliminate padding waste |
| **SJF Alignment** | Short sequences complete first | Form "length clusters" |

> **Typical Alignment Process**:
> ```
> t=0: [50, 52, 55, 60, 100] → Select length 50
> t=1: [51, 52, 55, 60, 100] → Select length 51
> t=2: [52, 52, 55, 60, 100] → Select length 52 (two sequences aligned)
> ...
> t=8: [60, 60, 60, 60, 100] → Four sequences perfectly aligned!
> ```

### 7. Tensor Parallelism (NEW ⭐)

Supports multi-GPU distributed inference, breaking single-GPU memory limits:

- **Strategy**: Column Parallel + Row Parallel
  - MLP: Gate/Up projection Column Parallel, Down projection Row Parallel
  - Attention: QKV projection Column Parallel, Output projection Row Parallel
- **Communication**: Uses `all_reduce` to synchronize activations
- **Advantage**: Supports super large model deployment while maintaining efficient inference

---

## 📊 Performance Benchmark

### Three-Way Comparison · Single-User Throughput (L20 / Qwen3-0.6B)

Fair comparison against vLLM and nano-vllm under fully aligned conditions (short context is micro-vllm's sweet spot):

> **Hardware**: NVIDIA L20 &nbsp;|&nbsp; **Model**: Qwen3-0.6B (bf16) &nbsp;|&nbsp; **Input**: 8 tokens &nbsp;|&nbsp; **Output**: 200 tokens &nbsp;|&nbsp; **Sampling**: temperature=0.01 &nbsp;|&nbsp; **Method**: median of 7 runs, each engine on a dedicated GPU

| Framework | Throughput (tokens/s) | Relative |
|:-----|:----------------:|:--------:|
| **micro-vllm** | **405.8** | **1.21×** |
| vLLM 0.21.0 | 386.4 | 1.15× |
| nano-vllm | 335.8 | 1.00× |

- micro-vllm leads vLLM by **+5.0%** and nano-vllm by **+20.9%** in single-user throughput
- All three engines show < 1 token/s standard deviation across 7 runs — stable and reproducible
- Edge comes from hand-written CUDA GEMV + CUDA Graph amortizing kernel fixed overhead at M=1 decode

### Three-Way Comparison · Batch Throughput (L20 / Qwen3-0.6B)

As concurrency rises, the bottleneck shifts from kernel launch overhead to memory bandwidth and tensor-core GEMM, where vLLM's inductor compilation pays off:

> **Hardware**: NVIDIA L20 &nbsp;|&nbsp; **Model**: Qwen3-0.6B (bf16) &nbsp;|&nbsp; **Input**: 128 tokens &nbsp;|&nbsp; **Output**: 256 tokens &nbsp;|&nbsp; **Sampling**: temperature=0.01

| Concurrency | micro-vllm | vLLM 0.21.0 | nano-vllm |
|:------:|:----------:|:-----------:|:---------:|
| 1      | **405.8**  | 386.4       | 335.8     |
| 32     | 8,090      | **8,597**   | 7,238     |
| 64     | 10,716     | **13,256**  | 11,672    |

- At bs=1 micro-vllm leads vLLM by **+5.0%**; as concurrency grows vLLM overtakes via compiled tensor-core GEMM
- micro-vllm's positioning is clear: **low-concurrency, latency-sensitive** serving (single-user / few-user interactive scenarios), not high-concurrency aggregate throughput

### Continuous Batching · 1000 Requests (L20 / Qwen3-0.6B)

1000 mixed requests (max_tokens 40-80 random, temp=0.6, ignore_eos), all enqueued then drained — the realistic high-concurrency serving scenario:

> **Hardware**: NVIDIA L20 &nbsp;|&nbsp; **Model**: Qwen3-0.6B (bf16) &nbsp;|&nbsp; **Method**: 3-run stable values

| Framework | Throughput (tok/s) | Steps |
|:-----|:----------------:|:-----:|
| **micro-vllm** | **28,110** | 130 |
| nano-vllm | 27,638 | 153 |

- micro-vllm leads nano-vllm by **+1.7%**; per-step GPU time at bs=512 is also lower (14.33ms vs 14.36ms)
- Recent wins: sampler `reduce-overhead` removal (saves a 155MB logits DtoD copy, 410us/step), QK-Norm+RoPE single-kernel fusion, `prepare()` dirty-flag (steady-state CPU 0.88ms→0), final-norm fusion
- Single-batch (same prompt, 500 out, full prefill+decode): bs=32 **7,060** vs 6,465 (+9.2%), bs=64 **9,864** vs 9,332 (+5.7%)

Benchmark scripts live in [`benchmark/`](benchmark/README.md).

---





## 🚀 Quick Start

### Installation

```bash
# Clone the project (recommend switching to online branch for latest optimizations)
git clone https://github.com/benyuereal/micro-vllm.git
cd micro-vllm
# Install dependencies
pip install -r requirements.txt
```

### Model Download

```bash
huggingface-cli download --resume-download Qwen/Qwen2-7B-Chat \
  --local-dir ~/huggingface/Qwen2-7B-Chat/ \
  --local-dir-use-symlinks False
```

### Basic Usage

```python
from core.engine import InferenceEngine

# Initialize engine
engine = InferenceEngine(
    model_path="/path/to/Qwen2-7B-Chat",
    max_batch_size=32
)

# Batch generation
results = engine.generate(
    ["Hello", "AI is"],
    max_tokens=100
)
for prompt, text in results.items():
    print(f"{prompt}: {text}")

# Streaming generation
for token, text in engine.stream_generate("The future of AI is", max_tokens=50):
    print(text, end="", flush=True)
```

### Start API Server

```bash
python api_server.py
```

After startup, available at:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## 🌐 API Reference

### Non-streaming Generation

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Java file upload code",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### Streaming Generation

```bash
curl -X POST "http://localhost:8000/generate_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a SpringBoot file upload code",
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": true
  }'
```

---

## ⚖️ Comparison Test

### Start vLLM Server

```bash
python -m vllm.entrypoints.openapi.api_server \
    --model /path/to/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --served-model-name Qwen3-0.6B
```

### Test Request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3-0.6B",
        "prompt": "Hello, write a Java file upload code",
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": true
    }'
```

---

## 📦 Project Structure

```
micro-vllm/
├── core/
│   ├── engine.py           # Inference engine entry
│   ├── scheduler.py        # Continuous batching scheduler
│   ├── cache_manager.py    # PagedAttention KV cache manager
│   ├── paged_attention.py  # Paged attention implementation
│   ├── sequence.py         # Sequence state management
│   └── layer/
│       ├── model_graph.py  # CUDA Graph wrapper
│       └── sampler.py      # torch.compile sampler
├── models/
│   └── qwen_adapter.py     # Qwen model adapter
├── kernel/
│   ├── rmsnorm.py          # RMSNorm custom implementation
│   └── swiglu.py       # ⭐ SwiGLU fused activation (Latest optimization)
├── api_server.py           # FastAPI server
└── requirements.txt         # Project dependencies
```

---

## 📋 Dependencies

- torch >= 2.0.0
- transformers >= 4.56.0
- flash-attn >= 2.0.0
- fastapi >= 0.100.0

---

## 💡 Note

This framework is suitable for small-to-medium scale LLM service production deployment, achieving 105% of vLLM's single-user throughput with clean code that is easy to understand and extend.
---

## 📄 License

MIT License
