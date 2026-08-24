# 优化进展记录

## 1000 请求连续批处理吞吐（L20 / Qwen3-0.6B / bf16）

目标：1000 请求批次吞吐超过 nano-vllm。

| 阶段 | 1000 请求吞吐 | 领先 nano |
|:-----|:------------:|:---------:|
| 会话起点 | 22,770 | -18% |
| 4 项融合+Python 优化 | 28,110 | +1.7% |
| update_sequences 快速路径 | 28,865 | +4.5% |
| Gumbel 采样 kernel | **30,316** | **+9.7%** |

累计 **+33.1%**，5 个 commit 全部 push（`41cc2a3` / `beb5da3` / `79e4e73` / `f57af14` / `f6a836a`）。

### 各阶段优化内容

**1. 4 项融合+Python 优化（22,770 → 28,110，commit 41cc2a3）**

差分 profiling（micro vs nano 逐 kernel 对比 bs=512 decode）定位 micro 每步 GPU 慢 626us 的三个根因：

- **采样器去 `reduce-overhead`**（sampler.py）：`torch.compile(mode="reduce-overhead")` 给 sampler 单独捕获 CUDA Graph，每步需把 [bs,vocab]=155MB logits 拷进静态图输入 buffer（3 次 DtoD = 410us/步）。nano 用默认编译无此拷贝。改默认模式后 410us→1us
- **QK-Norm + RoPE 单 kernel 融合**（kernel/rotary.py）：原 prerope 路径 `qk_norm_inplace` + `apply_rope_decode` 两趟读写 head_dim，新 `_qk_norm_rope_kernel` 单 kernel 内 RMSNorm + half-split RoPE 原地写回，542→350us/步
- **`prepare()` 脏标志**（context.py + engine.py）：稳态 decode 每步 batch 成员/顺序不变，原每步重建 512 元素 cur_ids/ctx_lens 列表 + 比较 = ~1.2ms CPU。改 engine 维护 `_ctx_batch_dirty`，仅序列完成/prefill 新进/append 跨 block 时置脏，稳态 CPU 0.88ms→0
- **final_norm 融合**（model_graph.py）：decode 末尾 `rmsnorm_` 直写 `_hidden[:bs]`，替代 HF 原生 RMSNorm 的 bf16↔fp32 中间拷贝

**2. update_sequences decode 稳态快速路径（28,110 → 28,865，commit 79e4e73）**

每步 15.19ms wall vs GPU 14.33ms → 每步 ~0.86ms CPU 串行段，其中 0.75ms 是 `update_sequences` 的 512 seq Python 循环（rank0()/dict.get()/方法调用 ~10 操作/seq）。无流式 client、无 stop 串、全 decode seq 时走快速路径（只做 append+position+finished 判断）；先扫条件再应用防 double append；padded batch 含循环复制重复 seq，append 与 mark_finished 均去重。发现慢路径条件立即回退完整路径。

**3. Gumbel-max 采样单 Triton kernel（28,865 → 30,316，commit f57af14）**

bs=512 per-kernel profile：sampler 930us（top_p=1.0 走 torch.compile，`logits.float()` 物化 311MB fp32 + softmax + exponential，HBM ~1.4GB/步）。Gumbel-max 定理 `argmax(logits/temp + gumbel_noise)` 无需显式 softmax：新 `kernel/sampling.py` 两趟 kernel（19 chunk 局部 max + 跨 chunk reduce），bf16 直读 + fp32 寄存器归约，HBM 仅 logits 读一遍（155MB），1225→269us/步（2.6x）。随机数用 xorshift-multiply hash（比 tl.rand Philox 快），seed 每步递增保证噪声 i.i.d.

### 剩余空间评估

bs=512 每步 14302us 分解：flash 6450us(45%) + GEMM 6211us(43%) 占 88%，flash 与 nano 同款 kernel、GEMM 已验证 HBM 极限；CPU 侧已榨干（每步 gap <0.1ms）。1000 请求场景 bf16 下已接近上限，继续单点优化 ROI 很低。

## 单用户 decode 优化（#39）：flash-decoding + paged-KV off-by-one 修复

目标：单用户长上下文 decode 追平/超过 vLLM。

### 排查：短/长上下文为何 micro 掉速

重测三方对比时发现 micro 短上下文（128 in/256 out）400 tok/s、长上下文（256 in/768 out）掉到 361 tok/s，而 vLLM 几乎持平（386→385）。差分 profiling 定位：

- **flash attention kernel**：短上下文 546us/步 → 长上下文 967us/步（+77%）。KV cache 随 seqlen 线性增长，attention 读整个 KV，成本随之上升——符合物理预期
- **GEMV（权重读）**：2229→2008us/步，与上下文无关，完全持平

根因在 `models/qwen3/adapter.py` 的 `num_splits`：bs=1 时写死 `num_splits=1`（注释"短 KV 无需 split"）。长上下文下这导致 flash-decoding 的 split-KV 并行失效——只有 16 个 CTA（16 q_heads）在跑，L20 有 92 个 SM，**SM 利用率仅 17%**，KV 读的 in-flight load 喂不饱 HBM。vLLM 用 flash-decoding 按 seqlen 动态 split，bs=1 也并行到全部 SM，所以不掉速。

### 修复 1：bs=1 开 flash-decoding（auto split-KV）

`num_splits=1 if bs==1` → `num_splits=0 if bs==1`（0 = flash 按 seqlen 自动选 split）。

| 上下文 | 修复前 | 修复后 | 提升 |
|:-------|:------:|:------:|:----:|
| 短 128/256 | 400 | 411 | +2.8% |
| 长 256/768 | 361 | **410** | **+13.6%** |

bs>1 路径未动（只改 bs==1 的 1→0），1000 请求吞吐无回归（30,316→30,280，噪声内）。greedy 对齐 HF（MATCH True）。

### 修复 2：paged-KV off-by-one（真实 engine bug）

长上下文测试触发 illegal memory access。定位：`core/cache_manager.py` 的 `alloc()` 在 **prefill 长度恰为 block_size(256) 整数倍**时，最后一块 `_pos = n_tokens % block_size = 0`（应为 block_size，因为该块已写满）。首 decode `append()` 误判该块有空位、复用 slot 0 且不分配新块 → block_table 少一列 → flash 读 `block_table[1]=-1` 越界。

- 触发条件：prefill=256/512/768…（255/257 正常）
- 修复：`last_pos = n_tokens % self.block_size or self.block_size`
- 验证：256/256、512/100、256/768 全通过，greedy 对齐 HF（MATCH True）

### 三方对比最终数据（L20 / Qwen3-0.6B / bf16）

**批次吞吐（128 in / 256 out，temp=0.01）**

| 并发数 | micro-vllm | vLLM 0.21.0 | nano-vllm |
|:------:|:----------:|:-----------:|:---------:|
| 1      | **409.1**  | 386.2       | 340.9     |
| 32     | 7,503      | **7,749**   | 6,438     |
| 64     | 10,469     | **11,547**  | 9,635     |

**单用户长上下文（256 in / 768 out，7 轮中位数）**

| 框架 | 吞吐 (tok/s) |
|:-----|:-----------:|
| **micro-vllm** | **410.4** |
| vLLM 0.21.0 | 385.4 |
| nano-vllm | 347.1 |

bs=1 micro 领先 vLLM +6.0%（批次）/ +6.5%（长上下文）；并发增大后 vLLM 凭 inductor 编译 + tensor-core GEMM 反超。micro 定位：低并发延迟敏感场景。

## 下一步

TileRT 权重读∥计算 overlap（persistent kernel），理论单用户 410 → 679 tok/s。当前 GEMV 已近 HBM 极限，2-3x 需 persistent kernel 吃 kernel 间 HBM round-trip + 权重读/计算 overlap。
