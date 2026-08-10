# TileRT 改造方案：DeepSeek MLA attention 全融合 TileLang kernel

> 分支：tilert | 基准见 [tilert-baseline.md](tilert-baseline.md) | TileRT 文章要点见 memory

## 一、目标与依据

### profile 数据（execution gap 在哪）
attention 内部各阶段占比（eager, bs=8, max_len=1024, 每层每步）：

| region | us/层 | %attn | 是否落 HBM |
|---|---|---|---|
| store | 47.5 | 5.1% | 写 1 个 latent |
| gather | 75.7 | 8.2% | 读 [bs,1024,576] 写 HBM |
| **kvb** | **379.8** | **40.9%** | rmsnorm+GEMM，写 [bs,1024,16,256] 落 HBM |
| **rope** | **360.0** | **38.8%** | RoPE+cat+pad，写 [bs,1024,16,192]×2 落 HBM |
| flash | 43.5 | 4.7% | 真正 attention |
| oproj | 21.4 | 2.3% | output proj |

**核心结论**：kvb + rope = 79.7%，是 execution gap 主体。这两步把 latent `[bs,1024,576]` 经 GEMM 展开 + RoPE 拼接成 `[bs,1024,16,192]`（k/v），全部写回 HBM，再被 flash 读回——巨大 memory round-trip。flash 本身只占 4.7%。

### 改造目标
用一个 TileLang kernel 融合 `gather → kv_b_proj → RoPE → score → softmax → ·V`，让 kvb/rope 的中间张量 `[bs,1024,16,256]` **留在 smem/register，不落 HBM**，直接喂给 attention 计算。理论上吃掉 attention 内 79.7% 的大部分。

## 二、当前数据流（改造前）

```
cache: k_cache [n_blocks, block_size, 1, 576]  (latent, paged)
  │
  ├─ store: 新 token latent 写入 slot
  │
  ├─ gather: k_flat[slots] → latents [bs, 1024, 576]          ← 落 HBM
  │
  ├─ kvb:  latents → split → rmsnorm(ckv[512]) → kv_b_proj
  │        → kv [bs,1024,16,256] → k_nope[128]|v[128]         ← 落 HBM (40.9%)
  │
  ├─ rope: q_pe/k_pe RoPE + cat
  │        → k_full [bs,1024,16,192], v_fa [bs,1024,16,192]   ← 落 HBM (38.8%)
  │
  ├─ flash: flash_attn_varlen_func(q, k_v, v_v) → attn_out    (4.7%)
  │
  └─ oproj: linear(attn_out, o_w) → out
```

中间张量 `[bs,1024,16,256]`、`[bs,1024,16,192]` 反复在 HBM 进出，是 gap 根源。

## 三、改造后数据流（TileLang 融合 kernel）

```
输入（GPU tensor, graph-friendly）:
  Q       [bs, 16, 128]   (q_nope, 已算好)
  Q_pe    [bs, 16, 64]    (q_pe, 已 RoPE 或 kernel 内 RoPE)
  k_cache [n_blocks, block_size, 1, 576]   (latent, paged, 不展开)
  kvb_w   [4096, 512]     (kv_b_proj 权重, 16*(128+128) × 512)
  kva_ln_w, kva_ln_eps    (rmsnorm)
  block_table [bs, max_seq_blocks]
  cache_seqlens [bs]
  cos/sin [max_pos, 64]   (RoPE table, interleaved)

kernel 内部（每个 tile 处理一段 KV）:
  for kv_tile in tiles(over 1024 positions):
    1. block_table 跳读: latents_tile = k_cache[block_table[tile]]   ← smem [block_N, 576]
    2. split: ckv_tile = latents_tile[:, :512], k_pe_tile = latents_tile[:, 512:]
    3. rmsnorm: ckv_tile = rmsnorm(ckv_tile)                          ← fragment
    4. kv_b_proj: kv_tile = ckv_tile @ kvb_w.T  → [block_N, 16, 256]  ← fragment (GEMM)
    5. split: k_nope_tile = kv_tile[..., :128], v_tile = kv_tile[..., 128:]
    6. RoPE: k_pe_tile = rope(k_pe_tile, cos[tile_pos], sin[tile_pos])← fragment
    7. score: s = (Q @ k_nope_tile + Q_pe @ k_pe_tile) * scale        ← fragment [block_H, block_N]
    8. online softmax: 更新 m, s_sum, acc_o                           ← fragment
    9. ·V: acc_o += softmax(s) @ v_tile                               ← fragment [block_H, 128]
  输出: O [bs, 16, 128]  (attn_out, 无 seq_len 维)
```

**关键**：步骤 4-9 全在 fragment/smem 流，`[block_N, 16, 256]` 中间量**不落 HBM**。
这同时解决了：
1. execution gap（kvb+rope 不落 HBM）
2. batch 串扰（kernel 按 cache_seqlens 每条 seq 只读自己的 KV，不 gather 成定长密集）
3. max_len 进形状（输出 `[bs,16,128]` 无 seq_len 维，桶上限可解除）
4. gather 本身也被吃进 kernel（block_table 跳读）

## 四、分阶段实施

### Phase 1：TileLang MLA decode kernel（attention 全融合）
- 写 `kernel/tile_mla_decode.py`，TileLang 实现。
- 输入：Q, Q_pe, k_cache(latent), kvb_w, ln 参数, block_table, cache_seqlens, cos/sin。
- 输出：O [bs, 16, 128]。
- 参考 `tilelang/examples/deepseek_mla/example_mla_decode_paged.py` 的 paged + split-kv 结构，
  但把 KV 输入从"已展开 dv"改成"latent 576 + kernel 内 kv_b_proj 展开"。
- 替换 adapter.py attention() 的 (2)~(6) 为这个 kernel 调用。
- store(1) 和 oproj(7) 暂留外面（store 是写 cache，oproj 是 attention 后的线性层，可后续融）。

### Phase 2：接入 CUDA Graph
- kernel 输入全是 GPU tensor（block_table/cache_seqlens/cos/sin 常驻），graph-friendly。
- 验证 graph capture 成功，bs 1/2/4/8/16/32/40 全 OK。
- 测吞吐 + 逐 token 延迟，对照基准。

### Phase 3：解除 1024 限制
- kernel 输出无 seq_len 维，attention 不再依赖固定 max_len。
- max_position 可恢复到模型真实值（4096）或更高。
- batch 串扰 bug 随 paged kernel 自然消除。

### Phase 4（远期）：persistent kernel / warp specialization
- 朝 TileRT 文章的常驻 Engine Kernel 演进，跨层融合。

## 五、技术风险与决策点

### 风险1：kv_b_proj 在 kernel 内做 GEMM 的效率
- kvb 是 `[block_N, 512] @ [512, 4096]` 的 GEMM，block_N 通常 32-128。
- TileLang 有 gemm tile op（`T.gemm`），smem→fragment 的 MMA 应能高效。
- 风险：小 block_N 下 GEMM 效率不如 cublas。但省掉的 HBM 往返远大于此。
- 决策：先实现，profile 对比。若 GEMM 成瓶颈，调 block_N 或用 split。

### 风险2：RoPE interleaved 在 kernel 内实现
- DeepSeek 用 interleaved RoPE，只作用 k_pe（64 维）。
- TileLang 需手写 interleaved rotate（even/odd 交错），cos/sin 从 table 读。
- 决策：kernel 内对 k_pe 做 RoPE，q_pe 仍在外（q 只 1 个位置，开销小）或也并入。

### 风险3：rmsnorm 在 kernel 内
- kv_b_proj 前的 rmsnorm(ckv[512]) 需在 kernel 内做（reduce + scale）。
- TileLang 有 reduce op，per-token rmsnorm 是 [512] 上的 reduce，可行。
- 决策：并入 kernel。

### 风险4：graph capture 与 TileLang JIT
- TileLang 用 `@tilelang.jit` 编译 kernel，编译产物是 CUDA kernel。
- 需确认编译后的 kernel 能被 CUDA Graph capture（应该是普通 kernel launch，可以）。
- 决策：Phase 2 验证。

## 六、验收标准
- DeepSeek 吞吐 > 72.2 tok/s（基准），目标提升 30%+（>94 tok/s）。
- DeepSeek 每步 decode 延迟 < 13.47ms（基准），目标 < 10ms。
- batch 串扰 bug 消除（不等长 batch 正确）。
- CUDA Graph capture 全 bs OK。
- 正确性：输出与改造前逐 token 一致（temperature=0 对比）。
