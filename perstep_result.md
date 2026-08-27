# per_step 优化结果（Qwen3.8-27B W8A16 + DFlash2 spec decode，GPU3）

基线 f877649（main HEAD），per_step ≈ 55.66ms。北极星 = 单用户 3000 输出稳态 tok/s。

## 1. profile 分解（CUDA event 差分法，300 步均值）

### 基线（无 int8，per_step = 54.08ms / 3000 e2e 55.61ms）

| 段 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| v.replay | 39.57ms | 73.2% | verify CUDA graph（64 层 Marlin int8 GEMM 主导） |
| d.fwd | 6.22ms | 11.5% | draft 5 层 forward（bf16） |
| d.select | 3.72ms | 6.9% | draft lm_head + topk + selector |
| v.lmhead | 3.53ms | 6.5% | verify lm_head（bf16，eager） |
| fillkv | 0.70ms | 1.3% | 增量 context KV 投影 |
| v.fill | 0.11ms | 0.2% | verify graph replay 前 buffer 填 |
| d.embed | 0.05ms | 0.1% | draft query embed |
| **accept** | **0.06ms** | **0.1%** | **argmax + 2 次 D2H + Python 贪心循环** |
| v.auxcopy / d.ctxkv | 0.02ms | 0.0% | aux 拷贝 / ctx_kv 补填 |
| other | 0.85ms | 1.6% | Python/调度/launch gap |

**关键发现：贪心接受段（任务候选 #1）只有 0.06ms（0.1%），不是瓶颈**——两次 D2H +
Python 循环已被 CUDA graph 吃掉 launch 间隙，无需 Triton 化。

### kernel 级（torch.profiler，20 步）

- Marlin int8 GEMM：35.98ms/step（68.9%）——verify 64 层 × 4 GEMM，256 次/step
- cutlass bf16 大 GEMM：7.06ms/step（13.5%）——lm_head ×2（verify+draft 共享）+ fc
- cutlass bf16 小 GEMM：5.68ms/step——draft 5 层 GEMM
- GDN recurrent 1.57ms、flash 0.22ms、其余小 op

## 2. 优化：补全 W8A16 量化（lm_head + draft 5 层 bf16→int8 Marlin）

模型本身已 W8A16（64 层 int8），**lm_head（quant ignore 列表存 bf16，2.54GB）+
draft 5 层（3.33GB）是仅有的 bf16 残留**。转 int8 是补全量化，非新近似。

### 关键 bug（导致早先误判）

`quantize_group128` 原 `round(wf / amax)` 值域 [-1,1]（int8 全 0/±1，反量化 127x
偏小）→ 改 `round(wf * 127 / amax)` 值域 [-127,127]。这个 bug 让 lm_head int8 早先
测得 acceptance 4.312→0.192 崩溃、draft int8 测得 4.312→0.000 崩溃——**都是假象**。
修复后两者 acceptance 均保持 4.312。

### 各优化前后（per_step / tok_s / acceptance）

| 配置 | per_step (3000 e2e) | steady_tok_s | acc | 首个分歧 |
|---|---|---|---|---|
| 基线（无 int8） | 55.61ms | 54.8 | 1.984 | @160 |
| draft int8 only | 51.90ms (profile) | — | 4.312 | @160 |
| lm_head int8 only | 50.65ms (profile) | — | 4.312 | @160 |
| **both int8（默认）** | **49.75ms** | **61.1** | **1.978** | **@160** |

- draft int8：d.fwd 6.22→4.03ms（-2.19ms）
- lm_head int8：v.lmhead 3.53→1.74ms + d.select 3.75→1.94ms（-3.43ms）
- 合计：per_step 55.61→49.75ms（**-10.5%**），steady_tok_s 54.8→61.1（**+11.5%**）

### 优化后 profile（both int8，per_step = 48.68ms）

| 段 | 耗时 | 占比 |
|---|---|---|
| v.replay | 39.58ms | 81.3% |
| d.fwd | 4.26ms | 8.8% |
| d.select | 1.94ms | 4.0% |
| v.lmhead | 1.74ms | 3.6% |
| 其余 | 1.16ms | 2.4% |

## 3. token 一致性

- **spec vs 非spec 首个分歧 @160，与基线完全相同**（非spec=4937 spec=271，逐 token
  一致）。这是 acc_result.md 记录的 target 侧 bf16 drift（verify M=8 vs decode M=1
  经 GDN fp32 递归累积放大），**非 int8 引入**——int8 前后分歧点、分歧 token 完全一致。
- ≤128 非决胜区逐 token 匹配（与基线一致）。
- acceptance 1.984→1.978（3000 e2e）/ 4.312（256 短测）保持，draft 提议质量未退化。
- 输出连贯（300 token 文本正常，无乱码/塌循环）。

## 4. 剩余空间（v.replay 39.58ms，81.3%）

verify GEMM 是 64 层 Marlin int8（24.33GB 权重/step）。L20 HBM 实测上限 ~652GB/s
→ 理论下限 24.33/0.652 ≈ 37.3ms，当前 39.58ms = **94% HBM 利用率，已到物理极限**。
自研 kernel 到不了 Marlin 级（memory 记录：1.10-1.12x 硬顶），persistent/融合对
memory-bound 段无 ROI（memory 记录多次验证）。**bf16 单卡无进一步 per_step 空间，
2x 必须 FP8（用户禁止）。**

## 5. commit 链（不 push）

```
7780a68 draft/lm_head int8 默认开（MICRO_DRAFT_INT8/MICRO_LMHEAD_INT8 默认 1）
f544299 lm_head int8(Marlin)：v.lmhead 3.53→1.74ms + d.select 3.75→1.94ms，per_step 54.08→50.65ms(-6.3%)
8a5d7d7 draft 5 层 int8(Marlin)：d.fwd 6.22→4.03ms，per_step 54.08→51.90ms(-4.0%)
f877649 (基线) merge: 重构 TP通信/warmup 抽独立文件 + SpecDecodeController→SpecEngine
```

## 6. 改动文件

- `kernel/marlin/__init__.py`：新增 `MarlinLinear` / `quantize_group128`（含 127x
  修复）/ `build_marlin_from_int8` / `linear_to_marlin`
- `models/qwen3_5/adapter.py`：`_LMHEAD_INT8` 开关 + prepare_weights 末尾 lm_head 转换
- `models/dflash/draft_model.py`：`_DRAFT_INT8` 开关 + `convert_to_int8()`（47 Linear）
- `core/engine.py`：`_build_spec_engine` 调 `draft_model.convert_to_int8()`

## 7. 测试脚本（/vllm-workspace/tmp/）

- `profile_perstep.py` / `profile_perstep2.py`：分段 profile
- `profile_perstep_kernels.py`：kernel 级 profile
- `bench_lmhead_int8.py`：lm_head bf16 vs int8 隔离 bench
- `test_lmhead_argmax.py` / `test_draft_sensitivity.py` / `test_draft_int8_err.py`：
  正确性/敏感度
- `bench_3000_perstep.py`：3000 稳态北极星
