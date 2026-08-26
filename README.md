# micro-vllm

<p align="center">
  <img width="300" src="assets/logo.png" alt="logo">
</p>

> A high-performance LLM inference engine built **from scratch** — PagedAttention, Flash Attention, CUDA Graph, continuous batching, hand-written CUDA/Triton/TileLang kernels, **W8A16 quantization**, **speculative decoding (DFlash2)**, and a **GDN (Gated DeltaNet) hybrid-attention** model stack. No vLLM/SGLang runtime dependency: the whole engine is ~8k lines of Python you can read end to end.
>
> 🚀 **Headline**: single-GPU **Qwen3.8-27B (W8A16) + DFlash2 speculative decoding** hits **101.5 tok/s = 1.77× vLLM** (57.4 tok/s) on an L20. Non-spec single-user long-context (Qwen3-0.6B) reaches **410 tok/s**, +6.5% over vLLM.

## ✨ Features

* 🚀 **Continuous Batching** — dynamic batch filling, ~99% GPU utilization at batch=32
* 💾 **PagedAttention** — KV-cache paging, ~5% fragmentation, no pre-allocation waste
* ⚡ **Flash Attention v2** — auto RoPE, zero-copy paged KV, flash-decoding (auto split-KV) for bs=1
* 🔥 **CUDA Graph** — whole-graph capture for decode; pre-captured across batch sizes
* 🎲 **Speculative Decoding (DFlash2)** — N=7 draft / M=8 verify, greedy accept, GDN state checkpoint-rollback; **1.77× vLLM** on Qwen3.8-27B
* 🧮 **W8A16 Quantization** — Marlin-format int8 group-128 weights, TileLang verify GEMM + hand-written int8 GEMV
* 🌊 **GDN Hybrid Attention** — Gated DeltaNet linear-attention layers (no KV cache, recursive state) mixed with full attention
* 🧠 **MLA + MoE** — Multi-head Latent Attention and Mixture-of-Experts (DeepSeek-V2-Lite)
* 🌐 **Tensor Parallelism** — column/row parallel, multi-GPU
* 📡 **OpenAI-Compatible API** — real token-level streaming SSE, `ignore_eos`, configurable context length
* 📖 **Clean Codebase** — ~8k lines of Python, easy to learn and extend

---

## 📚 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Core Technologies](#-core-technologies)
- [Supported Models](#-supported-models)
- [Speculative Decoding](#-speculative-decoding)
- [Performance Benchmark](#-performance-benchmark)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        InferenceEngine                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Scheduler  │───▶│  KVCacheMgr  │───▶│  ModelGraphRunner │  │
│  │(Continuous   │    │   (Paging)   │    │ (TP + CUDA Graph) │  │
│  │  Batching)   │    │              │    │                   │  │
│  └──────────────┘    └──────────────┘    └───────────────────┘  │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ModelAdapter (per-arch)  →  Flash Attention / GDN / MLA │    │
│  │  + W8A16 int8 GEMM/GEMV  +  SwiGLU / RMSNorm / RoPE      │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SpecDecodeController (DFlash2 draft-verify-accept)      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Responsibility |
|-----------|----------------|
| `InferenceEngine` | Engine entry: model load, scheduling, KV-cache, execution |
| `Scheduler` | Continuous batching, dynamic fill |
| `KVCacheManager` | PagedAttention KV-cache paging |
| `ModelGraphRunner` | CUDA Graph capture and replay (decode) |
| `ModelAdapter` | Per-architecture forward (Qwen / Qwen3 / Qwen3.5 / DeepSeek) |
| `SpecDecodeController` | DFlash2 draft → target verify → greedy accept |
| `Sampler` | torch.compile token sampler |

---

## 🔬 Core Technologies

### 1. PagedAttention

KV-cache paging (Block=256 tokens), dynamic allocation, ~5% fragmentation, no pre-allocation waste.

```python
cache_manager.alloc(seq_id, num_tokens)  # Allocate cache blocks
cache_manager.append(seq_id)             # Append new token
cache_manager.free(seq_id)               # Free cache
```

### 2. Flash Attention v2

`flash_attn_with_kvcache` for efficient attention: auto RoPE, zero-copy paged KV update, and **flash-decoding** (auto split-KV) that keeps KV reads parallel across all SMs at bs=1 as context grows.

### 3. W8A16 Quantization

Marlin-format int8 weights (group-128, byte-128 encoding) with two decode paths:

* **Verify GEMM** (`kernel/gemm_int8_triton.py`) — TileLang int8 GEMM for the fixed M=8 speculative-verify shape
* **int8 GEMV** (`kernel/gemv_int8.cu`) — hand-written CUDA for M=1 decode

`lm_head` stays bf16; int8 dequant is a temporary compute, never persisted.

### 4. GDN (Gated DeltaNet) Hybrid Attention

Qwen3.5 / Qwen3.8 mix **linear-attention GDN layers** (no KV cache — only a per-seq recursive fp32 state + short conv state) with periodic **full-attention** layers (`full_attention_interval=4`). GDN decode updates only real rows of the state pool; the delta-rule recurrence runs in fp32.

### 5. MLA + MoE (DeepSeek)

Multi-head Latent Attention (compressed KV latent) and Mixture-of-Experts routing for DeepSeek-V2-Lite, with TileLang paged-MLA and routed-expert GEMM kernels.

### 6. CUDA Graph

Decode forward is captured as a single CUDA Graph and replayed per step, pre-captured across batch sizes to eliminate inter-kernel launch overhead.

### 7. Tensor Parallelism

Column-parallel (QKV / Gate-Up) + row-parallel (O / Down) with `all_reduce` on activations, breaking single-GPU memory limits.

---

## 🧩 Supported Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| Qwen2-7B-Chat | dense, GQA | reference baseline |
| Qwen3-0.6B | dense, GQA + QK-Norm | primary benchmark model |
| Qwen3 | dense, GQA + QK-Norm, independent head_dim | HF-named weights |
| Qwen3.5 | **GDN hybrid** (linear + full attn) | 1-centered RMSNorm, partial RoPE |
| Qwen3.8-27B | **GDN hybrid + W8A16** | spec-decoding headline model |
| DeepSeek-V2-Lite | **MLA + MoE** | latent attention + routed experts |

Model adapters live in `models/` (`qwen/`, `qwen3/`, `qwen3_5/`, `deepseek/`); the DFlash2 draft model in `models/dflash/`.

---

## 🎲 Speculative Decoding

DFlash2 draft-verify-accept, greedy:

* **Draft** — a 5-layer sliding-window (2048) non-causal model proposes N=7 tokens in one forward, fed by the target's intermediate hidden states (aux layers) projected through an FC + norm.
* **Verify** — the target runs one causal forward over M=1+N=8 tokens (fixed shape).
* **Accept** — greedy match: accept while draft==target-argmax, bonus = target's prediction at the first mismatch. Single-sequence output is token-identical to non-spec greedy.

**GDN state rollback** is the correctness crux: verify advances the GDN recursive/conv state over all 1+N tokens, but only `accepted` are valid. Verify enables per-token GDN state checkpoints (zero overhead in normal prefill) and rolls back to `checkpoint[accepted]`.

```python
engine = InferenceEngine(
    model_path="/path/to/Qwen3.8-27B-W8A16",
    spec_decode=True,
    draft_model_path="/path/to/dflash2-draft",
    num_speculative_tokens=7,
)
res = engine.generate_spec_decode("prompt", max_tokens=3000)
```

---

## 📊 Performance Benchmark

### Speculative Decoding · Single-GPU (L20 / Qwen3.8-27B W8A16 + DFlash2)

| Framework | Throughput (tok/s) | Relative |
|:-----|:----------------:|:--------:|
| **micro-vllm** | **101.5** | **1.77×** |
| vLLM (TP1, DFlash2) | 57.4 | 1.00× |

Edge comes from the W8A16 int8 verify GEMM/GEMV + CUDA Graph amortizing launch overhead at the fixed M=8 verify shape.

### Single-User Long-Context (L20 / Qwen3-0.6B bf16)

256 in / 768 out, temperature=0.01, median of 7 runs, each engine on a dedicated GPU:

| Framework | Throughput (tok/s) | Relative |
|:-----|:----------------:|:--------:|
| **micro-vllm** | **410.4** | **1.18×** |
| vLLM 0.21.0 | 385.4 | 1.11× |
| nano-vllm | 347.1 | 1.00× |

### Batch Throughput (L20 / Qwen3-0.6B bf16)

128 in / 256 out, temperature=0.01:

| Concurrency | micro-vllm | vLLM 0.21.0 | nano-vllm |
|:------:|:----------:|:-----------:|:---------:|
| 1      | **409.1**  | 386.2       | 340.9     |
| 32     | 7,503      | **7,749**   | 6,438     |
| 64     | 10,469     | **11,547**  | 9,635     |

At bs=1 micro-vllm leads vLLM by +6.0%; as concurrency grows vLLM overtakes via compiled tensor-core GEMM. micro-vllm's positioning: **low-concurrency, latency-sensitive** serving.

### Continuous Batching · 1000 Requests (L20 / Qwen3-0.6B bf16)

1000 mixed requests (max_tokens 40–80, temp=0.6, ignore_eos), all enqueued then drained:

| Framework | Throughput (tok/s) | Steps |
|:-----|:----------------:|:-----:|
| **micro-vllm** | **30,316** | 130 |
| nano-vllm | 27,638 | 153 |

Benchmark scripts and load-test commands live in [`benchmark/`](benchmark/README.md).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/benyuereal/micro-vllm.git
cd micro-vllm
pip install -r requirements.txt
```

### Basic Usage

```python
from core.engine import InferenceEngine

engine = InferenceEngine(
    model_path="/path/to/Qwen3-0.6B",
    max_batch_size=32,
)

# Batch generation
results = engine.generate(["Hello", "AI is"], max_tokens=100)
for prompt, text in results.items():
    print(f"{prompt}: {text}")

# Streaming generation
for token, text in engine.stream_generate("The future of AI is", max_tokens=50):
    print(text, end="", flush=True)
```

### Start API Server

```bash
# Non-spec (continuous batching)
python api_server.py --model /path/to/Qwen3-0.6B

# Speculative decoding (DFlash2, greedy)
python api_server.py \
    --model /path/to/Qwen3.8-27B-W8A16 \
    --spec-decode \
    --draft-model /path/to/dflash2-draft \
    --num-spec-tokens 7 \
    --max-context-length 4096
```

Server args: `--model` / `--model-name`, `--spec-decode`, `--draft-model`, `--num-spec-tokens`, `--max-batch-size` (default 512), `--max-context-length` (default 1024), `--served-model-name`. After startup:

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## 🌐 API Reference

OpenAI-compatible endpoints: `POST /v1/completions`, `POST /v1/chat/completions` (both support `stream: true` with real token-level SSE), `GET /v1/models`. Legacy: `POST /generate`, `POST /batch_generate`, `POST /generate_stream`.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B",
    "prompt": "Hello, write a Java file upload code",
    "max_tokens": 1000,
    "temperature": 0,
    "ignore_eos": true,
    "stream": true
  }'
```

> The spec-decode path is **greedy-only** (`temperature=0`); sampling requests return HTTP 400 on a spec-enabled instance.

---

## 📦 Project Structure

```
micro-vllm/
├── core/
│   ├── engine.py           # Inference engine entry
│   ├── scheduler.py        # Continuous batching scheduler
│   ├── cache_manager.py    # PagedAttention KV cache manager
│   ├── paged_attention.py  # Paged attention
│   ├── sequence.py         # Sequence state
│   ├── spec_decode.py      # DFlash2 draft-verify-accept controller
│   ├── model_loader.py     # Weight loading (incl. W8A16 unpack)
│   └── layer/
│       ├── model_graph.py  # CUDA Graph wrapper
│       ├── model_prefill.py# Prefill runner
│       ├── rope.py         # RoPE
│       └── sampler.py      # torch.compile sampler
├── models/
│   ├── qwen/  qwen3/  qwen3_5/  deepseek/   # per-arch adapters
│   └── dflash/                        # DFlash2 draft model
├── kernel/
│   ├── gemv.cu / gemv.py               # hand-written CUDA GEMV
│   ├── gemv_int8.cu / gemv_int8.py     # int8 GEMV (W8A16)
│   ├── gemm_int8_triton.py             # TileLang verify int8 GEMM
│   ├── mla.py / moe.py / pre_mla.py    # DeepSeek MLA + MoE
│   ├── rmsnorm.py / rotary.py / sampling.py
│   └── dense_mlp.py / quant.py
├── api_server.py           # FastAPI / OpenAI-compatible server
├── demo/                   # correctness + profiling scripts
├── benchmark/              # throughput / load-test scripts
└── requirements.txt
```

---

## 📋 Dependencies

- torch >= 2.0.0
- transformers >= 4.56.0
- flash-attn >= 2.0.0
- fastapi >= 0.100.0
- tilelang (verify int8 GEMM; Triton fallback)

---

## 💡 Note

This framework targets small-to-medium scale LLM serving with clean, readable code: single-user latency (spec decode, W8A16) and low-concurrency interactive workloads. For high-concurrency aggregate throughput, vLLM's compiled tensor-core GEMM currently leads.

## 📄 License

MIT License
