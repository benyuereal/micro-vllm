# micro-vllm 代码精简报告（cleanup-round3）

基线：f877649（14413 行 .py）→ 最终 440c174（10350 行 .py）
**总删除 4063 行 .py（28.2%）+ 992 行 .md 文档**

## 各批删除清单

### 批次1（commit 9a3bf16）早期 Qwen 适配 — 14413 → 13981（-432）
- 删 `models/qwen/`（166 行，Qwen1/2/2.5 GQA+SwiGLU）
- 删 `models/qwen3/`（258 行，Qwen3 非 3.5）
- `models/__init__.py`：去 qwen3 分支 + qwen 兜底改 `raise ValueError`
- `core/model_loader.py`：去 `qwen3`/`qwen3-0.6b` 死别名
- 验证：import api_server OK；build_adapter qwen3_5→Qwen3_5Adapter、deepseek→DeepSeekAdapter、qwen3→raise ValueError

### 批次2（commit 9145b52）冗余文档 — .py 行数不变
- 删 `.plan.md`(105) `TODO.md`(11) `UPDATE.md`(93) `docs/tilert-plan.md`(333) `docs/tilert.md`(291) `docs/tilert-baseline.md`(159)，共 992 行 .md
- 保留 README.md / README_zh.md
- 验证：import api_server OK

### 批次3（commit 18a3761）demo/ 精简到 4 个 — 13981 → 12305（-1676）
- 保留：`chat.py`(233) `verify_spec_qwen38.py`(75) `bench_w8a16.py`(44) `align_qwen38.py`(84)
- 删 20 个：profile_*(10) bench_*(5) dbg_*(2) baseline_qwen38.py test_w8a16.py align_qwen35.py
- 删前 grep 确认无 core/benchmark import demo 模块
- 验证：import api_server OK

### 批次4（commit 5a9f8a3）benchmark/ 精简到 4 个 — 12305 → 10477（-1828）
- 保留：`benchmark_single_user.py`(125) `benchmark1000_throughput.py`(114) `validate_spec_decode.py`(404) `benchmark_spec_decode.py`(96)
- 删 11 个：bench_draft_gemm benchmark_throuput benchmark_tp benchmark_tp_perf profile_tp_rank0 test_dflash2_draft test_prefix_cache test_prefix_cache_long test_tp_correctness validate_dflash2_wiring validate_draft_tilelang
- 保留脚本默认模型 `/models/Qwen3-0.6B`（qwen3 adapter 已删）→ `/models/Qwen3.8-27B-INT8-W8A16-MTP`（仅改 MODEL 默认路径，逻辑不变）
- 验证：import api_server OK

### 批次5（commit e736ba7）kernel/ 死代码 — 10477 → 10350（-127）
- 删 `kernel/dflash_ops.py`(15)：rope_half_split 仅被已删 qwen3 adapter 引用（validate_spec_decode.py 自带同名函数）
- 删 `kernel/draft_gemm.py`(112)：仅被已删 bench_draft_gemm/validate_draft_tilelang 引用
- 逐个 grep 验证其余 kernel 模块均被 core/models/dflash import，全部保留
- 保守保留：`kernel/marlin/generate_kernels.py`（codegen 工具，sm80_*.cu 由它生成）
- 验证：import api_server OK；qwen3_5/deepseek adapter 均 build 成功

### 批次6（commit 440c174）tp_comm 命名精简 — 行数不变（纯改名）
- `TPCommunicator`：broadcast_tokens→bcast_tokens、receive_tokens→recv_tokens、broadcast_batch→bcast_batch、broadcast_waiting→bcast_waiting、receive_batch→recv_batch
- 同步改 `core/engine.py` 的 `self._tp_comm.*` 调用点（5 处）
- engine 公开包装 `tp_broadcast_*/tp_receive_*` 保持不变（api_server 调用）
- 验证：import api_server OK；无残留 `_tp_comm.broadcast_*/receive_*` 引用；5 个新方法 + 5 个 tp_* 包装均存在

## Smoke test 结果（GPU5，Qwen3.8-27B W8A16）
1. **非 spec 32 token**：SMOKE_OK，1.3s，文本连贯（"The field was founded by John McCarthy..."）
2. **spec decode 验收**（demo/verify_spec_qwen38.py OUT_TOK=32）：
   - 非 spec 32 tok / 1.27s = 25.2 tok/s
   - spec 32 tok / 0.27s = 119.3 tok/s，acceptance=6.750，steps=4
   - **PASS：spec 与非 spec greedy 前 32 个逐 token 一致**

## commit 链
```
440c174 精简批次6: core/tp_comm.py 方法命名精简 (broadcast_*/receive_* → bcast_*/recv_*)
e736ba7 精简批次5: 删 kernel/ 死代码 (dflash_ops.py + draft_gemm.py)
5a9f8a3 精简批次4: benchmark/ 精简到 4 个 (2567→739 行)
18a3761 精简批次3: demo/ 精简到 4 个 (2112→436 行)
9145b52 精简批次2: 删冗余文档 (.plan.md/TODO.md/UPDATE.md/docs/tilert*.md)
9a3bf16 精简批次1: 删早期 Qwen 适配 (models/qwen 166行 + models/qwen3 258行)
f877649 (基线)
```

## 备注
- 精简比例 28.2%（目标"约 30%"，6 个批次清单全部执行完；剩余文件均被 core/models/dflash/api_server 直接 import，保守不删）
- 核心文件（dflash/draft_model.py、qwen3_5/、spec_decode.py、engine.py、scheduler.py、model_loader.py、layer/、api_server.py）全部未动
- 未 push
