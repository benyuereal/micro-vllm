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
- [x] 整层 + attention + MoE profile 完成（eager, bs=8）
- [x] 方案设计（本文档）
- [x] 阶段 0 尝试：TileLang MoE grouped GEMV —— **见下方关键修正**

## 七、阶段 0 实测修正（2026-08-10）

### 关键发现 1：eager profile 高估了 graph 下的开销

eager profile（bs=8）显示 MoE 1055us/层、attention 860us/层。但 **CUDA Graph 已摊掉大部分 launch 开销**：
- graph 单层 MoE vs eager：1334us vs 1385us（bs=8），graph 只省 4%
- 即 graph 下 MoE 仍是 ~1334us/层（bs=8），launch 不是主因，**是 GEMV 本身的 HBM 带宽 + 串行**

### 关键发现 2：基准是 bs=1，MoE 不是最大头

基准 72.2 tok/s = 单请求 bs=1，13.47ms/step。graph 下 MoE 占比随 bs 变化：

| bs | graph 单层 MoE | ×24 层 | 占 step 比 |
|---|---|---|---|
| **1** | **185us** | **4.43ms** | **33%** |
| 2 | 421us | 10.11ms | 75% |
| 4 | 732us | 17.56ms | 130% |
| 8 | 1318us | 31.64ms | 235% |

**bs=1（基准场景）MoE 只占 33%，attention + 其他占 67%。** eager profile 的 "MoE 51%" 是 bs=8 数字，bs=1 下 MoE 不是瓶颈。

### 关键发现 3：TileLang fragment GEMV 比 Triton 慢

- Triton `tl.sum(x*w, axis=1)` 单 expert GEMV：~4.3us（6 expert 一起 25.5us）
- TileLang fragment `for i,j in T.Parallel: prod=W*X` + `reduce_sum`：12.4us/expert
- TileLang `alloc_reducer` 模式：59.5us/expert（更慢）
- TileLang `T.gemm` 要求 M%16==0，GEMV(M=1) 不能用
- **端到端实测**：TileLang MoE 接入 decode = **21.2 tok/s** vs baseline 71.5 tok/s（慢 3.4×）

### 根因

1. TileLang 在 M=1 GEMV 场景没有 Triton 的 SIMT reduction 优化
2. 我的 kernel 结构串行 K expert（`for k in T.serial(K)`），浪费并行性（Triton grid=K 并行）
3. 优化目标错位：盯着 eager+bs=8 的 MoE 51%，但基准是 graph+bs=1 的 MoE 33%

### 方向修正

**阶段 0（MoE 优先）的前提不成立**——bs=1 下 MoE 不是最大头，且 TileLang GEMV 比 Triton 慢。
应转向 **bs=1 下的真正瓶颈**：attention 的 kvb+rope（graph 下待测）+ 层间 HBM round-trip。

下一步：
1. 量 **bs=1 graph 下 attention 各段**真实占比（kvb/rope/flash/oproj）
2. 量 **bs=1 graph 下整层各段**（qkv/attention/ffn）占比
3. 据此重定阶段优先级——很可能 attention 全融合（阶段 1）才是 bs=1 的收益点
4. MoE 若要优化，需换 grouped GEMM（tensor core）而非 GEMV，且只在 bs≥4 有意义

### 保留的代码

- `kernel/tilelang_moe.py`：TileLang MoE kernel（正确性通过，性能未达预期），保留备查
- `USE_TILELANG_MOE=1` 开关，默认关闭（走 Triton 路径）
- 各 prof 脚本：prof_layer/prof_moe/prof_graph_moe/prof_moe_graph_only.py

## 七补、bs=1 graph 精确 profile + TileLang MLA 现成算子（2026-08-10）

> 脚本：`prof_bs1_graph.py`、`prof_bs1_attn_breakdown.py`、`prof_bs1_fine.py`、`prof_bs1_cat.py`、`bench_tl_mla.py`
> 方法：单段 CUDA graph capture + replay 计时（纯 GPU 时间，剔除 launch/同步噪声），与基准 graph 可比。

### bs=1 graph 整层各段（基准场景，决定性数据）

| 段 | us/层 | %MoE层 | 说明 |
|---|---|---|---|
| qkv | 23 | 6% | norm+q_proj+kv_a_proj |
| **attention** | **157** | **41%** | 见下表 |
| **ffn(MoE)** | **179** | **47%** | 见下表 |
| next_qkv | 23 | 6% | 下一层入口 |
| **MoE层总计** | **383** | 100% | 3 dense+24 MoE → 10.31ms / 基准 13.47ms = **77%** |

**关键修正**：之前"MoE 只占 33%、attention 是瓶颈"的判断来自有缺陷的 eager 单段测量。
graph 精确数据下，**attention(157us) 与 MoE(179us) 基本持平**。两者都是单层大头。

### attention 内部细分（157us，bs=1/seq=1024）

| 子段 | us | 性质 |
|---|---|---|
| store（写新 latent） | 19 | cache 写 |
| gather（读全部 latent） | 15 | cache 读 |
| **kvb**（rmsnorm+kv_b_proj） | **47** | latent→[1,1024,16,256]，HBM 落盘 |
| rope（q_pe+k_pe） | 24 | 旋转位置编码 |
| **k_cat+k_reshape+v_pad** | **44** | flash 输入拼接，纯内存搬运 |
| flash | 8 | 真正的 attention 计算 |
| oproj | 9 | 输出投影 |

→ **attention 157us 里 flash 真正计算只有 8us**，其余 ~150us 全是 latent 展开/RoPE/拼接的 HBM
round-trip + launch 开销。这正是 TileRT 文章说的 execution gap。

### MoE 内部细分（179us，bs=1）

| 子段 | us |
|---|---|
| gate+topk | 9 |
| shared SwiGLU（2 shared expert 合并） | 58 |
| routed（6 expert） | 117 = gate_up 25 + down 34 + 中间循环开销 58 |

→ routed 真 GEMV 59us，另 58us 是 6 expert 逐 token 循环的 launch/HBM round-trip。
shared（58us）是固定开销，数据流简单，适合融合。

### 关键发现 4：TileLang 自带 paged MLA decode 算子，L20 可跑

`/models/tilelang/examples/deepseek_mla/example_mla_decode_paged.py`：
- **paged KV cache**：直接读 `block_table[batch, blk_idx]*block_size + offset`，和 micro-vllm 的 cache 逻辑一致
- persistent kernel：`T.Kernel(sm_num)` + `T.sync_grid()` 跨 SM 归约（`example_mla_decode_persistent.py`）
- online softmax + split-KV + logsumexp combine
- **正确性在 L20 + V2-Lite 维度通过**（`All close`，atol/rtol=0.01）

V2-Lite 维度（H=16, dv=128, dpe=64, d=192, fp16）实测：

| bs | seq | TileLang graph (us) |
|---|---|---|
| 1 | 1024 | **24.1** |
| 1 | 256 | 8.1 |
| 2 | 1024 | 24.3 |
| 4 | 1024 | 24.4 |
| 8 | 1024 | 24.4 |
| 16 | 1024 | 24.5 |

**batch 扩展性极好**：bs=1→16 几乎不变（grid 按 (batch, H//BLOCK_H) 划分，bs=1 时 16 个 head tile
打不满 92 SM，加 batch 用上空闲 SM）。

### 关键发现 5：现成 MLA kernel 只做 flash，不含 kvb/RoPE

现成 kernel 读的是 **已 split 的** `KV[..., :dv]`（k_nope/v）和 `K_pe[..., dpe]`（k_pe）。
**kvb 展开（latent→k_nope/v）和 RoPE 仍需在 kernel 外做**，或并入 kernel。
当前 attention 的 157us 里，kvb(47)+rope(24)+cat/pad(44) = **115us 是这个 kernel 没覆盖的**。
→ 全融合的真正工作 = 把 kvb+RoPE 并入这个 paged MLA persistent kernel，让 latent→k/v→flash 全在片内。

### 维度/layout 差异（接入要改的）

| 项 | TileLang 示例 | micro-vllm 现状 | 改动 |
|---|---|---|---|
| d / dv / dpe | 576/512/64（V2/V3） | 192/128/64（V2-Lite） | 参数化已支持 |
| block_size | 64 | 256 | kernel 要求 block_size≥block_N 且整除；改 block_N 或 cache block_size |
| KV layout | [b*max_seqlen_pad, h_kv, d] flatten | [n_blocks, block_size, 1, 576] | flatten + block_table 索引逻辑对齐 |
| dtype | fp16（写死） | bf16 | kernel dtype 参数化 |
| RoPE | 不含 | q_pe/k_pe interleaved | 需并入 kernel |

### 方向再修正（基于本次数据）

1. **attention 全融合是 bs=1 的最大收益点**：157us 里 8us 是计算，融合可吃掉 ~115us 的 kvb/rope/拼接 gap。
2. **现成 TileLang paged MLA 是地基**：flash 部分已有高性能实现（24us @ bs=1，且 batch 可扩展），
   不用从零写。工作重心 = 把 kvb+RoPE 融进去 + 接入 micro-vllm 的 paged cache（block_size 256 / bf16）。
3. **MoE 放第二阶段**：179us 里 shared(58us) 数据流简单可融；routed(117us) 有 data-dependent 路由，
   且 TileLang GEMV 用不了 tensor core（M%16 限制），需 grouped GEMM 思路，复杂度高。
4. 单层全融合最终形态：norm → [fused kvb+rope+paged-MLA+oproj] → norm → MoE，仍可分两个 kernel
   （attention 融一个、MoE 一个），host 每层 2 launch，比现在 278 launch/层 已是质变。

## 八、原始阶段规划（保留参考，已据上方修正）
