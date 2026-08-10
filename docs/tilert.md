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
测量脚本 `bench_tl_mla_e2e.py`（decode 默认走 TileLang 融合路径）。

### 做了什么

把 attention 的 `gather → rmsnorm → kv_b_proj → RoPE → cat/pad → flash` 六步压进**一个 TileLang persistent kernel**
（`kernel/mla.py`），中间的 `[bs,1024,16,256]` 全程不落 HBM：

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

## 三、已落地：融合 MoE decode kernel（routed experts）

### 成果

在融合 MLA 之上叠加 MoE routed-experts 融合，端到端再 +3.5%：

| | 吞吐 (bs=1, 200 token) | median step |
|---|---|---|
| baseline（flash + Triton MoE） | 74.8 tok/s | 13.40 ms |
| TileLang 融合 MLA | 83.7 tok/s | 11.87 ms |
| **TileLang 融合 MLA + MoE** | **86.6 tok/s** | **11.47 ms** |
| | **+15.8% vs baseline** | **-14.4%** |

端到端正确性：同 prompt 同 temp=0，baseline 与 MLA+MoE 双融合路径**逐 token 完全一致**（事实问答 "北京" 等均一致）。
测量脚本 `bench_tl_mla_e2e.py`（decode 默认走 TileLang 融合路径）。

MoE 段微基准分解（单层，K=6, E=64, INTER=1408, H=2048）：

| 段 | TileLang | Triton | 说明 |
|---|---|---|---|
| gate+topk | 8.8 us | 8.8 us | 留 PyTorch（占比小，graph-friendly） |
| routed experts | **85.3 us** | 106.1 us | 2-kernel M=16 融合，1.24x |
| shared expert | 57.8 us | 57.8 us | 留 PyTorch 大 GEMM（下阶段） |
| **MoE 全段** | **155.5 us** | 177.5 us | **1.14x，省 22us** |

### 做了什么

把 routed experts 的 `gate_up → silu·up·w → down` 三步压进**两个 TileLang kernel**
（`kernel/moe.py`），act 经 L2 暂存为 `[N, K, 16, INTER]`：

- **gu_silu**：grid=(N, K, cdiv(INTER,64))，每 block 算一个 (token, expert) 的 64 列 act。
  gate/up 各一次 `T.gemm`，silu 后写 `act16[n, kid, kid, :]`。
- **down**：grid=(N, cdiv(H,64))，每 token 一个 block，**串行 K 个 expert**，M=16 `T.gemm`
  算每个 expert 的输出，在 fp32 fragment 里累加，最后覆盖写。

### 核心难点：bs=1 GEMV 不能直接用 tensor core

DeepSeek-V2-Lite MoE 在 bs=1 decode 下每个 expert 都是 GEMV（M=1）。TileLang `T.gemm`
走 tensor-core `mma.h`，硬性要求 **`M % 16 == 0`**（`M must be divisible by 16, but got 1`）。

**解法：M=16 零填充 + grid 沿 K 维并行**。把 M=1 pad 成 M=16（15 行零填充无害），
让 grid 沿 top-K 维并行吃掉 padding——每个 block 算一个 (token, expert)，真实 act 在 16 行的第 `kid` 行。
这样能用上 tensor core，又不浪费（K=6 个 block 并行覆盖 6 个 expert）。

### 核心难点：三个 TileLang 精度/竞争陷阱

| 陷阱 | 现象 | 解法 |
|---|---|---|
| **bf16 silu 精度丢失** | `T.exp(-g)` 在 bf16 下偏差 ~1.5-1.9x，rel=1.26 完全错 | gate/up 的 shared buffer 用 **fp32**，silu 在 fp32 下算 → rel=0.012 |
| **gate weight 乘两次** | gu_silu 乘 `wk`，down 又乘 → 输出偏大 | 只在 gu_silu 乘一次，down 不乘 |
| **全局 `+=` 非原子 + 输出未清零** | `O[i] += val` 跨 expert block 竞争；atomic 版依赖输出清零但 jit 自动分配不保证 | 参照 TileKernels `reduce_fused`：**每 token 一个 block，block 内串行 K expert，fp32 fragment 累加，最后覆盖写**，彻底不用 atomic |

第三点最关键：第一版用 `T.atomic_add` 跨 block 累加（grid 沿 K 并行），在随机数据下碰巧 rel=0.003，
但真实 topk 数据 rel=3.89（输出未清零 + 竞争）。TileKernels 的 `reduce_fused` 给出正解——
**累加在 block 内的 fragment 做，不在全局 tensor 做**。

---

## 四、已落地：MLA 前置全融合（pre-MLA fusion）

### 成果

在 MLA+MoE 双融合之上，把 attention kernel **之前**的零碎 PyTorch 算子（q_proj、kva_proj、
store latent、rope(q_pe)、einsum absorb）融进 3 个 TileLang kernel，再 +8.7%：

| | 吞吐 (bs=1, 200 token) | median step |
|---|---|---|
| baseline（flash + Triton MoE） | 74.8 tok/s | 13.40 ms |
| TileLang 融合 MLA | 83.7 tok/s | 11.87 ms |
| TileLang 融合 MLA + MoE | 86.6 tok/s | 11.47 ms |
| **+ MLA 前置全融合** | **94.1 tok/s** | **10.54 ms** |
| | **+25.8% vs baseline** | **-21.3%** |

（原始复现基准 72.2 tok/s → 94.1 = **+30.3%**。）
端到端正确性：`1+1=`→`2`、`2+3=`→`5`、`Hello`→`, I am a 16 year`、英文续写合理，
逐 prompt 输出不同（temp=0）。集成路径 inline 对比 maxdiff q_nope=0.0 / q_pe=0.031 / A=0.016（bf16 精度内）。
测量脚本 `bench_tl_mla_e2e.py`。

### 做了什么

把 pre-MLA 的 ~7 个 PyTorch op 压成 **3 个 TileLang kernel**（`kernel/pre_mla.py`），rmsnorm 保留 Triton：

- **pre_qkv**：`q_proj` GEMM + rope(q_pe) epilogue。grid=(bs, 48)，每 block 算 q 的 64 输出列，
  `(nblk%3)==2` 的 block（q_pe 列）在 epilogue 做 rope。输出 `[bs, H, 16, 192]`。
- **pre_kva**：`kva_proj` GEMM + store epilogue。grid=(bs, 9)，epilogue 直写 paged cache
  `K_cache[blk_id, offset, 0, col]`，消除独立 store launch。
- **absorb**：`q_nope @ kvb_w_kn_t → A[bs, H, 512]`，复用已验证的 M=16 per-head GEMV。

为什么是 3 个 kernel 而非 1 个：q_proj 与 absorb 是**串行 GEMM**（absorb 需 q_proj 输出 q_nope），
TileLang 非 persistent kernel 无法跨 block 同步。评估过"融进每个 MLA split（4× 冗余）"——
q_proj 100M MACs ×4 = 545M，是 attention loop 的 122×，反而更慢。故拆成紧耦合 pre-kernel + absorb。
M=16 零填充（mma.h 要求 M%16==0），只读 row 0 真实数据，rows 1-15 恒零（GEMM 行独立，无害）。

### 核心难点：RoPE 输出必须是 deinterleaved 布局

HF DeepSeek 的 RoPE 用 `view_as_complex` 做**复数乘法**（interleaved 输入对 `(x[2k], x[2k+1])`），
adapter 的 `_apply_rope` 是其等价实数展开——但内部把 `x` 重新赋值为 **deinterleave 后**的张量再
`x*cos + rotate_half(x)*sin`，故**输出是 deinterleaved 布局**：位置 k 存旋转后的 `qpe[2k]`、
位置 k+half 存旋转后的 `qpe[2k+1]`。

第一版 kernel 误用 interleaved 输出（`out[j]=qpe[j]*cos[j]+rh[j]*sin[j]`），isolation 测试碰巧通过
（用了错误的 manual 参照），但集成后 q_pe maxdiff=20、decode 退化成重复 token（`1+1=`→`2222`）。
**正解**：直接做复数乘的实数展开，写 deinterleaved 输出：

```
for k in 0..half-1:
    a, b = qpe[2k], qpe[2k+1]            # interleaved 输入对
    out[k]      = a*cos(θ_k) - b*sin(θ_k)  → 写列 128+k
    out[k+half] = a*sin(θ_k) + b*cos(θ_k)  → 写列 128+k+half
```

cos/sin 全宽 `cat(freqs, freqs)`，故 `cs[k]==cs[k+half]`，只需 `cs[k]`。修复后 maxdiff=0.031。

### 其他集成细节

- **DeepSeek q_proj/kva_proj 无 bias**：TileLang kernel 要求非 None bias，`prepare_weights` 里
  对 None bias 分配零张量。
- **rmsnorm strided 输出**：Triton `rmsnorm_` 直接写 `_x16[:,0,:]`（stride=16*H），省一次 pad copy。
- **absorb 输出丢 M-pad 维**：输出 `[bs*H, kv_lora]`（不是 `[bs*H, 16, kv_lora]`），reshape 后
  `[bs, H, kv_lora]` contiguous，stride[1]=512 符合 MLA kernel 输入要求。
- **block_table 2D 索引**：`K_cache[blk_id, offset, 0, col]`（blk_id=block_table[b, pos//256],
  offset=pos%256），区分 max_seq_blocks（block_table 列数=4）与 n_blocks（cache 第一维=81）。

---

## 五、当前形态

每层 decode 是 **rmsnorm(Triton) + pre_qkv + pre_kva + absorb + fused MLA + gu_silu + down**。
gate/topk 和 shared expert 仍走 PyTorch。attention 段从 ~7 个 PyTorch op 压成 4 个 TileLang kernel，
MoE routed 段 2 个融合 kernel。pre-MLA 和 attention 内部两段 execution gap 各吃掉一截。

---

## 六、下一步

- **single-kernel MoE（吃掉 32us 间隙）**：gu_silu 28us + down 25us 单独跑只要 53us，
  两 kernel 串行 85us，中间 32us 是 launch/sync 间隙。用 `T.sync_grid()` 两阶段 persistent kernel
  把 gate_up→silu→down 融进单 kernel，act 全程在 L2，目标 ~55us（再省 30us/层）。
- **shared expert（58us）**：固定大 GEMM，无路由，可并入 MoE kernel 或单独融合。
- **单层全融合 persistent kernel** = norm → [pre-MLA + fused MLA + oproj] → norm → MoE，host 每层 1～2 launch。
- Qwen 同款适配。
