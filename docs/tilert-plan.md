# TileRT 改造方案：单层全融合 TileLang persistent kernel

> 分支：tilert | 基准见 [tilert-baseline.md](tilert-baseline.md) | TileRT 文章要点见 memory
> 范围：把**一整层 decode** 压成**一个 TileLang persistent kernel**，不是只融 attention。

## 一、目标与依据

### 确认的范围

用户确认："**单层全融合 persistent kernel**"。即把一整层 decode（norm + attention + oproj
+ norm + MoE）压成**一个 TileLang persistent kernel**，层内 h 状态在 smem/register/L2 流，
host **每层只 launch 一次**。27 层各自一个 persistent kernel。
- 不是只融 attention（太窄，attention 只占 41.7%）；
- 不是整个模型一个 kernel（27 层一次 launch，超出单层 scope，且捕获/调试代价过高）。

### profile 数据（execution gap 在哪）

整层 decode（eager, bs=8, max_len=1024, 每层每步，`prof_layer.py`+`prof_moe.py`）：

| 段 | us/层 | %层 | 内部最大头 |
|---|---|---|---|
| qkv | 5.6 | 0.3% | norm + q_proj + kv_a_proj |
| **attention** | **860** | **41.7%** | kvb 380 + rope 360 = 740（86%在attention内） |
| **ffn(MoE)** | **1161** | **56.3%** | gemv_loop 1055（90%在MoE内） |
| next_qkv | 34 | 1.7% | 下一层入口 |
| **层总计** | **2061** | 100% | **278 kernel 边界/层** |

**两个 execution gap 主体（占整层 87%）**：
1. **MoE gemv_loop 1055us（51%）**：8 token 各调 2 个 Triton kernel = 16 次 launch + HBM round-trip。
2. **attention kvb+rope 740us（36%）**：latent→[bs,1024,16,256] 写 HBM 再读回。

每层 **278 个 kernel 边界**。CUDA Graph 摊掉了 launch 开销，但**没省 kernel 间的 HBM
round-trip**（一个 kernel 写 HBM，下一个再读）。全融合的收益 = 消掉层内中间张量的 HBM 落盘。

### 核心约束
- **实现语言：TileLang**（不用 Triton）。当前 MoE 的 `grouped_gemv.py` 是 Triton，必须用 TileLang 重写。
- DeepSeek + Qwen 都要支持（本方案先做 DeepSeek，Qwen 复用框架，dense FFN 比 MoE 简单）。
- 保持固定 1024 上下文、热路径零架构分支。

## 二、当前数据流（改造前，一层 decode）

```
h [bs,2048]                                          (层输入, HBM)
 │
 ├─ compute_qkv(首层)/compute_next_qkv(后续层)
 │   ├─ rmsnorm_(h)                                   ── HBM round-trip
 │   ├─ q  = F.linear(h, q_w)        [bs,1920]       ── GEMM, 落 HBM
 │   └─ kva = F.linear(h, kv_a_w)    [bs,576]        ── GEMM, 落 HBM
 │      split → q_a[bs,512], q_pe[bs,64], kv_latent[bs,512]
 │      cache 写 slot
 │
 ├─ attention (逐层)
 │   ├─ store   新 latent → slot                       ── HBM
 │   ├─ gather  k_flat[slots] → [bs,1024,576]          ── HBM (大读)
 │   ├─ kvb     rmsnorm + kv_b_proj → [bs,1024,16,256] ── GEMM, 落 HBM (380us, 41%attn)
 │   ├─ rope    q/k RoPE + cat → k[bs,1024,16,192]     ── HBM (360us, 39%attn)
 │   ├─ flash   flash_attn_varlen → [bs,16,128]        ── 落 HBM (44us, 5%attn)
 │   └─ oproj   F.linear → [bs,2048]                   ── GEMM, 落 HBM (21us)
 │
 ├─ all_reduce (TP=1 时 no-op)
 │
 ├─ compute_ffn
 │   ├─ rmsnorm_residual                              ── HBM
 │   ├─ MoE (decode)
 │   │   ├─ gate    F.linear + softmax → [bs,64]       ── HBM (32us)
 │   │   ├─ topk    top-6 → idx[bs,6], w[bs,6]         ── HBM (10us)
 │   │   ├─ gemv_loop for i in bs:                     ── 16 次 Triton kernel (1055us, 90%MoE!)
 │   │   │    grouped_gate_up → silu*up*w → grouped_down
 │   │   └─ shared  x@shared_gu → silu*up → @shared_d  ── HBM (75us)
 │   └─ residual
 │
 ├─ all_reduce
 └─ → 下一层 h [bs,2048]
```

中间张量 `[bs,1024,16,256]`、`[bs,1024,16,192]`、每 token 的 `[K,2*inter]`/`[K,inter]`
全部写回 HBM 再被下一段读回——这是 278 个 kernel 边界的主要开销。

## 三、改造设计

### 总体：单层 persistent kernel

```
host: for layer in 27:
        launch single_layer_persistent_kernel(weights_l, h, kv_cache_l, ...)
              └─ 一个 kernel 跑完整层，层内 h/中间量在 smem/register/L2
```

- grid = SM 数（L20 = 48 SM），kernel 内 `for w in T.serial(waves)` 持续处理多个 tile（persistent）。
- 层内分阶段，阶段间用 `T.sync_grid()` / barrier 同步，中间量尽量不落 HBM。
- host 每层一次 launch（替代当前每层 ~278 次）。

### 难点与策略

**难点 1：MoE 是 data-dependent 路由**（topk 每 token 选不同 expert）
- 当前：逐 token for-loop，每 token 2 个 kernel（grouped_gate_up + grouped_down）。
- 全融合策略：参考 TileRT 文章的 **heterogeneous worker**——一个 persistent kernel 里，
  部分 warp/block 跑 gate→topk（产生 expert_idx），其余 worker 按 expert_idx 索引权重跑 GEMV。
  gate/topk 结果（`idx[bs,6]`, `w[bs,6]`）小，留 shared memory 或写一小块 HBM（L2 命中）。
  关键：**expert_idx 索引权重在 TileLang 里用 `tl.load(w_ptr + e*stride)` 风格的间接寻址**，
  和当前 Triton `_grouped_gate_up_kernel` 里 `e = tl.load(idx_ptr+pid_k)` 一致，TileLang 支持。
- bs=8 时 MoE 总 token-expert pair = 8×6 = 48，可一个 block 处理一个 token 的 6 个 expert，
  8 个 block 并行（L20 48 SM 远够）。

**难点 2：attention 的 kvb+rope 中间量巨大**（[bs,1024,16,256]）
- 这个张量 bs=8×1024×16×256×2B = 64MB，放不进 smem。
- 策略：**分 head 流式**。一个 block 处理一个 (batch, head) 的 flash 循环，kv_b_proj 只算该 head
  需要的 256 维切片，RoPE 在 register 里做，直接进 flash 的 QK·V 累加，**不写回完整 [bs,1024,16,256]**。
  即 kvb+rope 从"全展开再 flash"变成"按 head 边算边用"。
- kv_b_proj 权重 [512→4096] 按 head 16 等分（每 head 256 行），分块 load。

**难点 3：层间 h 仍要走 HBM**（单层 scope 内无法避免）
- h [bs,2048] = 8×2048×2B = 32KB，放不进跨 kernel 的 smem。
- 接受：层间 h 写 HBM（一次写一次读，2×32KB，~40us，占层 1.9%），这本来就是小头。
- 收益全在层内 278→1 个 kernel 边界。

### 阶段拆分（kernel 内部）

persistent kernel 内部按 tile 调度，逻辑分 4 段，段间 `T.sync_grid()`：

1. **QKV 段**：rmsnorm(h) → q_proj / kv_a_proj。h[bs,2048] 读一次，q/kva 算完留 register/smem。
2. **ATTN 段**：store latent → 按 head 流式 (gather切片 + kvb切片 + rope + flash + oproj切片)。
   oproj 输出 attn_out[bs,2048]。
3. **FFN 段**：rmsnorm_residual(h, attn_out) → gate/topk（data-dependent）→ MoE GEMV + shared → mlp_out[bs,2048]。
4. **RESIDUAL 段**：h = mlp_out + residual，写 HBM 给下一层。

### 实现语言映射（PyTorch → TileLang）

| 当前 PyTorch/Triton | TileLang 原语 |
|---|---|
| `T.Kernel(sm_num, threads=256)` + `for w in T.serial(waves)` | persistent kernel |
| `T.Pipelined(loop, num_stages=2)` | tile pipeline（data/compute overlap） |
| `T.sync_grid()` | 段间同步 |
| `T.gemm(shared, shared, fragment)` | 矩阵乘 |
| `T.copy(global, shared)` / `T.copy(fragment, global)` | HBM↔smem 搬运 |
| `T.alloc_shared` / `T.alloc_fragment` | smem / register |
| `tl.load(w + e*stride)` (Triton 间接寻址) | TileLang 同样支持指针算术间接寻址（MoE expert 索引） |
| `T.use_swizzle(10)` | bank conflict 优化 |
| warp specialization（ws 例） | data/compute warp 分离（MoE gate vs GEMV） |

## 四、分阶段实施

### 阶段 0：TileLang MoE grouped GEMV（先啃最大头 51%）

**为什么先做 MoE 而不是 attention**：MoE gemv_loop 1055us 是单层最大头（51%），且当前是
Triton（必须改 TileLang），结构相对独立（input [bs,2048] → output [bs,2048]），可单独验证。

- 用 TileLang 重写 `grouped_gate_up` + `grouped_down`，支持 expert_idx 间接寻址。
- 目标：把 16 次 kernel（8 token × 2）合成少量 TileLang kernel，中间 `act[K,inter]` 留 smem。
- 先 standalone 正确性（对齐 `moe.py` 输出），再接入 decode 验吞吐。
- **验收**：MoE 段 1055us → 目标 < 500us（消掉 launch + act 的 HBM round-trip）。

### 阶段 1：attention 全融合（kvb+rope 不落 HBM，36%）

- TileLang kernel：gather 切片 + kv_b_proj 按 head 切片 + RoPE(register) + flash + oproj。
- 参考 `examples/deepseek_mla/example_mla_decode_persistent.py` 的 persistent + split 结构，
  但**输入是 latent 不是展开 KV**——kv_b_proj 要进 kernel。
- **验收**：attention 段 860us → 目标 < 300us。

### 阶段 2：单层全融合（norm + attn + oproj + norm + MoE 一个 kernel）

- 把阶段 0+1 用 `T.sync_grid()` 串成一个 persistent kernel，层内 h 流 smem/register/L2。
- 处理 TP=1 no-op 的 all_reduce、residual。
- **验收**：整层 2061us → 目标 < 900us；端到端 72.2 tok/s → 目标 > 120 tok/s。

### 阶段 3：Qwen 适配 + 收尾

- Qwen 是 dense FFN（无 MoE），阶段 2 的 MoE 段换成单 MLP，更简单。
- 全量回归测试 + 基准对照（填 `tilert-baseline.md` 改造后表）。

## 五、风险与回退

- **TileLang 间接寻址/动态 shape 支持**：MoE expert_idx 是 runtime 值，需确认 TileLang 0.1.9
  支持指针间接寻址（Triton 支持，TileLang 需验证）。阶段 0 先 standalone 验证，不行则 MoE 段
  退回"gate/topk 在外 + TileLang GEMV 内"的半融合。
- **persistent kernel 的 smem 预算**：单层全融合要把多段 smem 复用，L20 smem 256KB/SM，
  需精打细算。阶段 2 先做正确再压 smem。
- **正确性**：每阶段 standalone 对齐 PyTorch 参考输出（rtol/atol），再接入。
- **回退**：每个阶段独立提交，性能不达预期可回退到上阶段。基线 0049aa8 始终可回。

## 六、当前状态

- [x] 基准记录（`tilert-baseline.md`）：DeepSeek 72.2 tok/s, 13.47ms/step；Qwen 45.9 tok/s, 21.19ms/step
- [x] 整层 + attention + MoE profile 完成
- [x] 方案设计（本文档）
- [ ] **待确认方向** → 然后开干
- [ ] 阶段 0：TileLang MoE grouped GEMV
- [ ] 阶段 1：attention 全融合
- [ ] 阶段 2：单层全融合
- [ ] 阶段 3：Qwen + 回归
