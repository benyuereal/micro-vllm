# TileRT 改造前性能基准

> 基准锚点：`tilert` 分支 HEAD = `0049aa8`（从 main 建出，固定 1024 上下文版本）
> 环境：NVIDIA L20 46GB，TP=1，CUDA Graph，bf16，固定 max_position=1024
> 测量：tokenizer 精确 token 计数（非 char/s），排除首轮热启动

## 测量方法

### 吞吐（/generate）
```python
t0 = time.time()
r = requests.post(url, json={"prompt": p, "max_tokens": 200, "temperature": 0.7}).json()
t1 = time.time()
gen = r["text"][len(p):] if r["text"].startswith(p) else r["text"]
gen_ids = tok.encode(gen, add_special_tokens=False)
tps = len(gen_ids) / (t1 - t0)
```
每 prompt 跑 3 轮，排除首轮热启动。

### 逐 token 延迟（/generate_stream）
POST `/generate_stream` stream=True，按 `data: {...}\n\n` 分割，记录每个 token 到达时间差。
- 注意 `min` 值有 asyncio.sleep 调度噪声（~0.05ms），看 `median`
- TTFT = 首个 token 时间 - 请求开始

### Server 启动
```bash
cd /models/micro-vllm
PORT=8001 nohup python3 api_server.py --model deepseek > /tmp/ds_base.log 2>&1 & disown
# qwen 同理：--model qwen
# kill 单独命令，避免 exit 144 中断
```

## 基准数据

### DeepSeek-V2-Lite

| 指标 | 值 |
|---|---|
| 吞吐（200 gen token，排除首轮） | **72.2 tok/s**（稳态 72.1–72.5） |
| 每步 decode 延迟 (median, 149 步) | **13.47 ms** |
| 每步 decode 延迟 (mean) | 13.37 ms |
| 每步 decode 延迟 (max) | 13.83 ms |
| TTFT | 109 ms |
| 稳态吞吐 (1000/median) | 74.3 tok/s |

吞吐明细（4 prompt × 3 轮，排除 p0 首轮）：
```
p0 run2: 72.1  p0 run3: 72.2
p1 run1: 72.1  p1 run2: 72.1  p1 run3: 72.2
p2 run1: 72.2  p2 run2: 72.2  p2 run3: 72.4
p3 run1: 72.4  p3 run2: 72.4  p3 run3: 72.5
```

### Qwen-7B-Chat

| 指标 | 值 |
|---|---|
| 吞吐（排除首轮） | **45.9 tok/s**（mean / median 一致） |
| 每步 decode 延迟 (median, 149 步) | **21.19 ms** |
| 每步 decode 延迟 (mean) | 21.05 ms |
| 每步 decode 延迟 (max) | 21.59 ms |
| TTFT | 68 ms |
| 稳态吞吐 (1000/median) | 47.2 tok/s |

## 关键观察

每步 decode 延迟（DeepSeek 13.47ms / Qwen 21.19ms）里大部分是 **execution gap**
（kernel 间 launch/sync/global memory round-trip），不是纯计算——这正是 tile op 要压缩的。

DeepSeek 的 attention 内部 `gather → kv_b_proj → RoPE → cat → flash_attn` 是多个独立 op，
中间 `[bs,1024,576]` / `[bs,1024,16,192]` 全写回 HBM，是 execution gap 最严重的部分。

## tile op 改造目标

- **实现语言：TileLang**（不用 Triton）
- **范围：单层全融合 persistent kernel**（不是只融合 attention）
  把一整层 decode（input_norm + q_proj + kv_a_proj + gather + kv_b_proj + RoPE + flash + o_proj
  + post_attn_norm + MoE/shared + residual）压成**一个 TileLang persistent kernel**，
  层内 h 状态在 smem/register/L2 流，host 每层只 launch 一次。27 层各自一个 persistent kernel。
- 详见 `docs/tilert-plan.md`。

## 整层 decode 各段耗时 profile

> `prof_layer.py` + `prof_moe.py`，eager，bs=8，max_len=1024，40 decode steps × 27 layers
> 一层 decode = qkv → attention → ffn → next_qkv（next_qkv 是下一层入口，近似算进层内）

| 段 | per_layer (us) | %层 | 说明 |
|---|---|---|---|
| qkv | 5.6 | 0.3% | input_norm + q_proj + kv_a_proj（首层） |
| **attention** | **860.5** | **41.7%** | 见下表细分 |
| **ffn (MoE)** | **1161.1** | **56.3%** | 见下表细分 |
| next_qkv | 34.2 | 1.7% | post_attn_norm + 下一层 q_proj + kv_a_proj |
| **层总计** | **2061** | 100% | 278 kernel 边界/层，7508 kernel/step |

### MoE 内部细分（`prof_moe.py`）

| region | per_layer (us) | %MoE | 说明 |
|---|---|---|---|
| gate | 31.7 | 2.7% | gate linear + softmax |
| topk | 10.4 | 0.9% | top-6 选 expert |
| **gemv_loop** | **1055.4** | **90.1%** | N=8 token × (grouped_gate_up + grouped_down) = 16 次 Triton kernel |
| shared | 74.5 | 6.4% | 2 个 shared expert 合并的 SwiGLU |
| **MoE 总计** | **1172** | 100% | |

### attention 内部细分（`prof_attention.py`）

| region | per_call (us/层) | %attn | 含义 |
|---|---|---|---|
| store | 47.5 | 5.1% | 写新 token latent 到 cache slot |
| gather | 75.7 | 8.2% | `k_flat[slots]` gather 成 [bs,1024,576] |
| **kvb** | **379.8** | **40.9%** | rmsnorm + kv_b_proj 展开 [bs,1024,16,256] |
| **rope** | **360.0** | **38.8%** | q/k RoPE + cat 拼接 + v pad |
| flash | 43.5 | 4.7% | flash_attn_varlen_func 真正的 attention |
| oproj | 21.4 | 2.3% | output projection |

**两个 execution gap 主体（占整层 87%）**：
1. **MoE gemv_loop 1055us（51%）**：8 token 各调 2 个 Triton kernel，16 次 launch + HBM round-trip。
   且当前是 **Triton**，改造必须用 TileLang 重写。
2. **attention kvb+rope 740us（36%）**：latent→[bs,1024,16,256] 写 HBM 再读回。

每层 278 个 kernel 边界 = 278 次 launch + 中间张量 HBM round-trip。
CUDA Graph 只省了 launch 开销，**没省 kernel 间的 HBM round-trip**——这才是 tile op 全融合的收益来源。

## attention 内部各阶段耗时 profile（原）

> `prof_attention.py`，eager 路径，bs=8，max_len=1024，40 decode steps × 27 layers
> CUDA event 手动计时，每层每步平均。eager 含 launch 开销，但各 op 相对占比 graph/eager 基本一致。

| region | per_call (us/层) | %attn | 含义 |
|---|---|---|---|
| store | 47.5 | 5.1% | 写新 token latent 到 cache slot |
| gather | 75.7 | 8.2% | `k_flat[slots]` gather 成 [bs,1024,576] |
| **kvb** | **379.8** | **40.9%** | rmsnorm + kv_b_proj 展开 [bs,1024,16,256] |
| **rope** | **360.0** | **38.8%** | q/k RoPE + cat 拼接 + v pad |
| flash | 43.5 | 4.7% | flash_attn_varlen_func 真正的 attention |
| oproj | 21.4 | 2.3% | output projection |

**关键发现**：
- `kvb` + `rope` 占 attention 的 **79.7%**，是 execution gap 的主体。
  这两步把 latent `[bs,1024,576]` 展开成 `[bs,1024,16,192]`（k）和 `[bs,1024,16,192]`（v），
  全部写回 HBM，再被 flash 读回——巨大的 memory round-trip。
- 真正的 attention 计算 `flash` 只占 4.7%。
- 印证 TileRT 文章：瓶颈不是"算得不够快"，而是 kernel 间的 memory round-trip（execution gap）。
- **tile op 融合的核心收益**：把 kvb+rope 的中间张量 `[bs,1024,16,256]` 留在 smem/register，
  不写回 HBM，直接喂给 attention 计算。理论上可吃掉 attention 内 79.7% 的大部分。

attention 每步总耗时(eager): 25.06 ms (27 层)，每层 0.928 ms。
（注：graph 下 launch 开销被省，attention 每步实际更低；基准 13.47ms/step 是 graph 全层含 MLP/MoE）

## 改造后对照

（tile op 完成后用同样方法测，填入此表对比）

| 指标 | 改造前 | 改造后 | 优化 |
|---|---|---|---|
| DeepSeek 吞吐 | 72.2 tok/s | TBD | TBD |
| DeepSeek 每步延迟 | 13.47 ms | TBD | TBD |
| Qwen 吞吐 | 45.9 tok/s | TBD | TBD |
| Qwen 每步延迟 | 21.19 ms | TBD | TBD |
