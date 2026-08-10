# TileRT 优化总结

> 把 TileRT 文章的 persistent-kernel / execution-gap 思想吸收进 micro-vllm，
> 针对 DeepSeek-V2-Lite（MLA + MoE）做单层全融合。本文记录已落地的成果与方法。
>
> 硬件：NVIDIA L20 46GB（sm_89, 92 SM, 100KB dynamic smem），TP=1，固定 1024 上下文，CUDA Graph，bf16。
> 基准分支：`tilert`（HEAD=0049aa8 改造前）。

---

## 一、问题：execution gap 是 bs=1 的真正瓶颈

bs=1 graph 精确 profile（单段 CUDA graph capture+replay，纯 GPU 时间）显示整层 383us：

| 段 | us/层 | 占 MoE 层 |
|---|---|---|
| qkv | 23 | 6% |
| **attention** | **157** | **41%** |
| **ffn(MoE)** | **179** | **47%** |
| next_qkv | 23 | 6% |

attention 157us 的内部细分才是关键：

| 子段 | us | 性质 |
|---|---|---|
| store / gather | 19 / 15 | cache 读写 |
| **kvb**（rmsnorm+kv_b_proj） | **47** | latent→[1,1024,16,256]，HBM 落盘 |
| rope | 24 | 旋转位置编码 |
| **k_cat+reshape+v_pad** | **44** | flash 输入拼接，纯内存搬运 |
| flash | 8 | 真正的 attention 计算 |
| oproj | 9 | 输出投影 |

**flash 真正计算只有 8us，其余 ~150us 全是 latent 展开 / RoPE / 拼接的 HBM round-trip + launch 开销。**
这正是 TileRT 文章说的 execution gap——算子间中间张量（`[bs,1024,16,256]`）反复写回 HBM 又读出。
融合的目标不是让 flash 更快，而是**吃掉这 115us 的搬运 gap**。

> 详见 `docs/tilert-plan.md` 第七节。profile 脚本：`prof_bs1_*.py`。

---

## 二、已落地：融合 MLA decode kernel（attention 全融合）

### 成果

| | 吞吐 (bs=1, 200 token) | median step |
|---|---|---|
| baseline（flash_attn_varlen） | 74.8 tok/s | 13.40 ms |
| **TileLang 融合 MLA** | **83.7 tok/s** | **11.87 ms** |
| | **+11.9%** | **-11.4%** |

复现基准 72.2 tok/s（本机 74.8，prompt 差异），融合后 **+11.9%**。
端到端正确性：同 prompt 同 temp=0 跑 136 token，baseline 与融合路径**逐 token 完全一致**。
测量脚本 `bench_tl_mla_e2e.py`，开关 `USE_TILELANG_MLA=1`。

### 做了什么

把 attention 的 `gather → rmsnorm → kv_b_proj → RoPE → cat/pad → flash` 六步压进**一个 TileLang persistent kernel**
（`kernel/tilelang_mla.py`），中间的 `[bs,1024,16,256]` 全程不落 HBM：

- **paged KV**：kernel 直接读 `block_table[batch, blk_idx]*block_size + offset`，和 micro-vllm 的 cache 逻辑一致。
- **online softmax + split-KV + logsumexp combine**：长上下文按 split 分摊到多 program，combine 阶段加权归并。
- **rmsnorm 分块**：latent 按 K_TILE=128 分块累加平方、`reduce_sum` 求 rinv，避免 `+=` 数据竞争。
- **RoPE 并入 kernel**：q_pe 由 adapter 旋转后传入，k_pe 在 kernel 内旋转（半分输出公式，与 HF `_apply_rope` 等价，PyTorch 验证 diff=0.0）。

### 核心难点 1：DeepSeek kv_b_proj 是 per-head 的

`kv_b_proj` shape = `[H*256, 512]`，输出 `[block_N, 4096] → view [block_N, H, 256]`，
**16 个 head 各有独立的 k_nope/v，不共享**。TileLang 自带的 `example_mla_decode_paged.py`
假设 KV 跨 head 共享（h_kv=1 广播），在真实 DeepSeek 下是错的——直接用会 rel_max≈1.0（完全错）。

**解法：MLA weight-absorption**，把 per-head 的 kvb 权重吸收进 Q / P，flash 循环内不再有 per-head KV：

1. 每层一次：`A[h] = Q_nope[h] @ kvb_w_kn[h]` → `A[H, kv_lora]`（吸收 k_nope 权重进 Q）
2. flash 循环内：`QK = A @ ckv_norm^T + Q_pe @ k_pe_rot^T`（标准 gemm，无 per-head KV）；
   softmax；`P += softmax @ ckv_norm`（累加到 `[H, kv_lora]` 空间）
3. 每层一次：`out[h] = P[h] @ kvb_w_v[h]^T`（post-multiply v 权重）

per-head 的两个小 einsum 各做一次，flash 内全是标准 gemm。详见
`tilelang-mla-perhead-kvb` 记忆 / `test_tl_mla_fused.py`、`test_tl_mla_long.py`。

### 核心难点 2：TileLang 编译期约束（L20 + bf16）

| 约束 | 现象 | 解法 |
|---|---|---|
| mma.h L99-100 | bf16 输入只注册 fp32 累加器，bf16 输出 fragment → static_assert | kvb gemm 用 fp32 累加器，写回时 cast bf16 |
| smem ≤ 100KB | 两份 32KB 权重 buffer 溢出 | 复用单个 weight shared buffer（kn 用完接 v）；ckv_s 复用存 k_pe；cos/sin 半宽 |
| GemmWarpPolicy.FullCol | block_N=32 时 N%64≠0 回退拆 M，warp_row_tiles=8<16 | 用 block_N=64（N=64 保 m_warp=1） |
| fragment 非均匀写 | `acc_p[i, kk*K_TILE+j] += ...` 跨切片写 fragment → `variable used before definition` | acc_p 改 shared buffer |
| fragment layout 推断冲突 | 额外的 rescale / P-accumulate 循环改变 acc_s layout，传播冲突到标量 fragment | `[BLOCK_H]` 标量状态（max/scale/sum/logsum）全部走 shared |
| 空 split NaN | loop_range=0 时 logsum=0 → acc_o/0=NaN 污染 combine | `acc_p = if(logsum>0, acc_p/logsum, 0)` |
| out_idx | 10 个参数 Output 在 index 9，写错指向 sin_k | `out_idx=[9]` |

---

## 三、当前形态

每层 decode 仍是两个 kernel：**fused MLA attention** + **MoE**（MoE 暂未融合，走原 Triton 路径）。
相比改造前的 278 launch/层，attention 段已压成 1 个融合 kernel。attention 的 execution gap 已吃掉一截。

---

## 四、下一步：MoE 融合

profile 数据（179us/层）：gate+topk 9us、shared SwiGLU 58us、routed 117us（真 GEMV 59us + 6 expert 逐 token 循环的 launch/HBM round-trip 58us）。

- **shared expert（58us）**：数据流简单，固定开销，适合先融合。
- **routed（117us）**：data-dependent 路由，6 expert 逐 token 循环的 launch 开销是大头；
  TileLang GEMV 用不了 tensor core（M%16 限制），需 grouped GEMM / persistent 循环思路，复杂度高。

最终目标：**单层全融合 persistent kernel** = norm → [fused MLA+oproj] → norm → MoE，host 每层 1～2 launch。
