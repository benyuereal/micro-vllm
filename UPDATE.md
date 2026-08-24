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

## 下一步：单用户 decode 优化（#39）

TileRT 权重读∥计算 overlap，理论 398 → 679 tok/s。
