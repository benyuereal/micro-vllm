"""DFlash2 投机解码控制器（Qwen3.8-27B W8A16 适配版）。

与 0.6B 版的核心差异（三大挑战的解法）：

1. GDN stateful 层（48 个 GDN 线性注意力层，递归+conv 有状态）：
   不能手写 forward（访问 self_attn 会崩），必须走 engine 的 adapter forward 路径
   （复用 self.model，权重已 prepare_weights）。控制器自己跑 forward 循环
   （embed → 逐层 adapter.prefill → final_norm → lm_head），与 ModelPrefillRunner 一致。

2. aux hidden state 收集：GDN 是顺序有状态的，不能 re-forward context（会二次推进
   递归状态）。故 aux 在主 forward 内收集——prefill/verify 每层 forward 后，若该层在
   target_layer_ids 里，就把 hidden state 写进滚动 aux_cache[ai, pos]。draft 的 context
   KV 由 aux_cache 投影（combine_hidden_states → precompute_context_kv）。

3. 显存复用 engine 模型：不 load 27GB 新副本，直接用 engine 的 self.model / adapter /
   prefill_runner / cache_manager。

GDN 状态回滚（投机解码正确性关键）：
   verify forward（1+N token）会把 GDN 递归/conv 状态推进过全部 1+N 个 token，但只有
   accepted 个有效。解法：verify 时开 GDN 逐 token 状态检查点（adapter 的 prefill kernel
   支持，CP_ENABLED constexpr，正常 prefill 零开销），接受后回滚到 checkpoint[accepted]。

KV cache（paged）：
   一次性 alloc(max_len) 个 slot，block_table 静态。verify 写 [kv_len-1, kv_len-1+1+N)，
   只 accepted 个有效，stale 区由下一步 verify 覆盖（自愈合）。flash 只读 cu_seqlens_k
   以内的有效区，不读 stale。

贪心投机解码等价性：draft 提议 N 个 token，target 一次 causal forward 验证，贪心接受
（draft==target argmax 则接受，首个不匹配处 bonus=target 预测）。单序列下与不开投机
解码的贪心输出逐 token 一致。
"""
import os
from typing import List, Optional

import torch

from models.base import PrefillMeta
from kernel.gemm_int8_triton import set_verify_gemm


class SpecEngine:
    """DFlash2 投机解码控制器（Qwen3.8 GDN 混合模型，复用 engine 模型）。"""

    def __init__(self, engine, draft_model, num_speculative_tokens: int,
                 mask_token_id: int, max_len: int = 1024):
        self.engine = engine
        self.model = engine.model                 # 复用 engine 的 W8A16 模型
        self.adapter = engine.adapter
        self.prefill_runner = engine.prefill_runner
        self.cache_manager = engine.cache_manager
        self.device = engine.device
        self.dtype = engine.dtype

        self.N = num_speculative_tokens
        self.mask_token_id = mask_token_id
        self.max_len = max_len

        # target 模块（经 adapter 访问，权重已 prepare_weights）
        self.embed = self.adapter.embed(self.model)
        self.blocks = self.adapter.blocks(self.model)
        self.final_norm = self.adapter.final_norm(self.model)
        self.lm_head = self.adapter.lm_head(self.model)
        self.num_layers = len(self.blocks)

        # target hidden size（aux_cache 维度）
        tc = getattr(self.model.config, "text_config", self.model.config)
        self.hidden = tc.hidden_size

        # aux 层（target_layer_ids）
        self.aux_layers = list(draft_model.target_layer_ids)
        self.num_aux = len(self.aux_layers)
        self.aux_index = {li: i for i, li in enumerate(self.aux_layers)}

        # draft
        self.draft = draft_model
        self.input_embedding_scale = draft_model.input_embedding_scale

        # 滚动 aux cache：[num_aux, max_len, hidden] bf16
        self.aux_cache = torch.zeros(
            self.num_aux, max_len, self.hidden, dtype=self.dtype, device=self.device)

        # 增量 context KV：draft attention 读的 context KV 由 aux 投影（combine→norm→
        # 各层 k/v proj + k_norm + RoPE）。旧实现每步对 [0, ctx_len) 全量重算 → 每步
        # O(ctx_len)，整段 O(n²)。改为常驻 buffer + 增量填充：每步只算本步 verify 新写入
        # aux 的那几个位置（anchor + accepted 个，a+1 个），hot path 纯切片读。
        # 形状 [num_draft_layers, max_len, num_kv_heads, head_dim] bf16。
        _d = self.draft
        self._ctx_k = torch.zeros(
            _d.num_layers, max_len, _d.layers[0].self_attn.num_kv_heads,
            _d.layers[0].self_attn.head_dim, dtype=self.dtype, device=self.device)
        self._ctx_v = torch.zeros(
            _d.num_layers, max_len, _d.layers[0].self_attn.num_kv_heads,
            _d.layers[0].self_attn.head_dim, dtype=self.dtype, device=self.device)
        # 已填充到哪个位置（_ctx_k/v 的 [0, done) 有效）。
        self._ctx_kv_done = 0

        # 静态 buffer（init 预分配，热路径零运行期分配：只 copy_/索引写）。
        # 每步 verify/draft 的 ids/positions/cu_seqlens/block_table 都从这些 buffer 取。
        M = 1 + self.N
        self._query_ids = torch.full((M,), self.mask_token_id,
                                     dtype=torch.int64, device=self.device)
        self._verify_ids = torch.empty(M, dtype=torch.int64, device=self.device)
        self._anchor_t = torch.empty(1, dtype=torch.int64, device=self.device)
        self._arange = torch.arange(max_len, dtype=torch.int64, device=self.device)
        self._cu_q = torch.empty(2, dtype=torch.int32, device=self.device)
        self._cu_k = torch.empty(2, dtype=torch.int32, device=self.device)
        self._prompt_buf = torch.empty(max_len, dtype=torch.int64, device=self.device)
        self._bt = torch.full((1, self.cache_manager.max_seq_blocks), -1,
                              dtype=torch.int32, device=self.device)

        # ---- verify CUDA graph 固定 buffer（M=1+N 固定 shape，replay 前 eager 填）----
        # vpos/vslot 是独立常驻 buffer（非 _arange/slot_mapping 的变偏移视图——视图 base
        # 指针每步变，graph replay 会读错地址）。replay 前 copy_ 进这两个固定 buffer。
        self._vpos = torch.empty(M, dtype=torch.int64, device=self.device)
        self._vslot = torch.empty(M, dtype=torch.int32, device=self.device)
        # aux 收集：graph 内写固定 _aux_tmp[ai]（目的地固定，graph 安全），replay 后
        # eager 拷到 aux_cache[ai, kv_len-1:...]（目的地偏移每步变，不能进 graph）。
        self._aux_tmp = torch.zeros(self.num_aux, M, self.hidden,
                                    dtype=self.dtype, device=self.device)
        # final_norm 输出（lm_head 之前）。graph 捕获到此为止，lm_head 在 replay 后
        # eager 跑（对齐非 spec decode：避免 [M,vocab] 大输出进 graph buffer）。
        self._vhidden = torch.empty(M, self.hidden, dtype=self.dtype, device=self.device)
        self._verify_model_graph = None

        # ---- draft CUDA graph 固定 buffer（与 verify graph 分开管理）----
        # draft 模型 forward（embed → 5 层 decoder → select_draft_tokens）单独 capture
        # 一个 graph。难点：draft attention 读 context KV [0:ctx_len]，ctx_len 每步增长
        # （变长），graph 需固定 shape。解法：固定读 [0:C]（C=max_len），用 attn_mask
        # 屏蔽 [ctx_len:C) 的无效 context 位置（exp(-inf)=0，softmax 结果与只读 [ctx_len)
        # 完全一致）。query 位置 / mask 都是固定 buffer，replay 前 eager 填。
        C = self.max_len
        self._dpos = torch.empty(M, dtype=torch.int64, device=self.device)
        # attn_mask 必须与 query 同 dtype（bf16）——SDPA 要求 bias dtype == query dtype。
        self._dmask = torch.zeros(M, C + M, dtype=self.dtype, device=self.device)
        self._draft_out = torch.empty(self.N, dtype=torch.int64, device=self.device)
        self._draft_model_graph = None

        # GDN 状态检查点 buffer（verify 时逐 token 存，接受后回滚）
        self._gdn_cp_state = None
        self._gdn_cp_conv = None
        self._alloc_gdn_checkpoint()
        # 去 rollback：上一步 verify 接受的 token 数。下一步 verify 的 GDN 初始状态
        # 直接读 checkpoint[accepted_prev]（省掉接受后 copy_ 回 pool 的 DtoD）。
        # None = 首步 verify（prefill 后，无检查点，初始状态从 pool 读）。
        self._gdn_accepted_prev = None

        # 统计
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

        # per-position 接受率诊断（MICRO_SPEC_POS_STATS=1 开启，默认关零开销）。
        # 按上下文长度分档（short<512 / mid 512-1500 / long>=1500，ctx_len=kv_len-1）：
        #   prefix[b][i] = 前缀接受到位置 i 的步数（accepted > i）
        #   match[b][i]  = draft[i]==target[i] 的步数（含前缀已断的步，参考用）
        self._pos_stats = None
        if os.environ.get("MICRO_SPEC_POS_STATS", "0") == "1":
            self._pos_stats = self._new_pos_stats()

    def _new_pos_stats(self):
        return {
            "steps": 0,
            "steps_b": {"short": 0, "mid": 0, "long": 0},
            "prefix": {b: [0] * self.N for b in ("short", "mid", "long")},
            "match": {b: [0] * self.N for b in ("short", "mid", "long")},
        }

    def _pos_bucket(self, ctx_len):
        if ctx_len < 512:
            return "short"
        if ctx_len < 1500:
            return "mid"
        return "long"

    def pos_stats_report(self):
        """per-position 接受率报告（诊断用）。返回 dict：各档 prefix/match 接受率曲线。"""
        if self._pos_stats is None:
            return None
        st = self._pos_stats
        out = {"steps": st["steps"], "buckets": {}}
        for b in ("short", "mid", "long"):
            nb = st["steps_b"][b]
            out["buckets"][b] = {
                "steps": nb,
                "prefix_rate": [st["prefix"][b][i] / nb if nb else 0.0
                                for i in range(self.N)],
                "match_rate": [st["match"][b][i] / nb if nb else 0.0
                               for i in range(self.N)],
            }
        return out

    # ------------------------------------------------------------------
    # GDN 检查点 buffer
    # ------------------------------------------------------------------
    def _alloc_gdn_checkpoint(self):
        """按 verify 最大 token 数 (1+N) 分配 GDN 状态检查点 buffer。
        recurrent: [M, n_gdn, H, DK, DV] fp32；conv: [M, n_gdn, K-1, conv_dim] bf16。"""
        n_gdn = self.adapter._n_gdn
        if n_gdn == 0:
            return  # 纯全注意力模型，无需 GDN 检查点
        M = 1 + self.N
        H, DK, DV = self.adapter._gdn_H, self.adapter._gdn_DK, self.adapter._gdn_DV
        conv_dim = self.adapter._gdn_conv_dim
        K = self.adapter._gdn_K
        self._gdn_cp_state = torch.zeros(
            M, n_gdn, H, DK, DV, dtype=torch.float32, device=self.device)
        self._gdn_cp_conv = torch.zeros(
            M, n_gdn, K - 1, conv_dim, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # GDN slot 管理（直接操作 adapter 的类级共享状态池）
    # ------------------------------------------------------------------
    def _gdn_pool(self):
        return self.adapter._shared[self.adapter._dev_key(self.device)]

    def _gdn_alloc(self):
        pool = self._gdn_pool()
        if not pool["free"]:
            raise RuntimeError("GDN 状态池耗尽")
        return pool["free"].pop()

    def _gdn_free(self, slot):
        self._gdn_pool()["free"].append(slot)

    # ------------------------------------------------------------------
    # target forward（走 adapter 路径，收集 aux，可选 GDN 检查点）
    # ------------------------------------------------------------------
    def _forward(self, input_ids, positions, slot_mapping,
                 cu_q, cu_k, block_table, max_sq, max_sk,
                 gdn_slot, collect_aux_from, gdn_checkpoint,
                 init_from_cp=False, init_state_s=None, init_state_c=None,
                 graph_mode=False):
        """跑一次 target 模型 forward（prefill/verify 共享本体），返回 logits [M, vocab]。

        调用方走两个阶段薄封装（见下方 _prefill/_verify），它们各自定死参数语义；
        本方法只认「跑 M 个 token」，不区分阶段。

        collect_aux_from: 从哪个绝对位置开始收集 aux（None=不收集），收集 M 个。
        gdn_checkpoint: 是否开 GDN 逐 token 检查点（verify 用）。
        init_from_cp/init_state_s/init_state_c: 去 rollback——非首步 verify 的 GDN
            初始状态直接读 checkpoint[accepted_prev]（init_state_s/c 是 [n_gdn,...]
            视图，token 索引已 bake 进 base 指针），省掉接受后 copy_ 回 pool 的 DtoD。
        graph_mode: verify CUDA graph 捕获/重放路径。True 时：
            - aux 写固定 _aux_tmp[ai]（目的地固定，graph 安全），不写 aux_cache
              （目的地偏移每步变，replay 后由 _verify_model_graph 拷过去）。
            - 只跑到 final_norm 写 _vhidden，不跑 lm_head（replay 后 eager 跑，
              对齐非 spec decode 避免 [M,vocab] 大输出进 graph buffer）。
            - 不做任何 host 侧 buffer 写（gdn_slot/init_idx 等由调用方在 capture
              前 eager 写好，capture 体内只读）。
        """
        M = input_ids.shape[0]
        h = self.embed(input_ids)

        # verify（M=1+N≈8）走双后端 int8 GEMM（TileLang 默认 / Triton 备选，
        # MICRO_VERIFY_GEMM 切换）：权重 HBM 只读一次（shared 内 dequant→bf16 +
        # GEMM），比原 CUDA tiled GEMV 快 ~12x（mlp_gu 3.59ms→0.29ms）。
        # prefill（M 大）本就走反量化 matmul，不受影响。
        set_verify_gemm(bool(gdn_checkpoint))

        # GDN 检查点开关（graph=prefill_runner 的 buffer）。prefill_runner 与 engine
        # 正常 prefill 路径【共享】，故必须在 finally 里复位 _gdn_cp_enabled=False，
        # 否则后续非 spec prefill（M 大）会往 8-token 检查点 buffer 写 M 个 → 越界。
        # graph_mode 下这些开关在 capture 前已 eager 设好且 capture 体内不再改，
        # finally 复位仍保留（capture 后 eager 路径依赖复位）。
        self.prefill_runner._gdn_cp_enabled = bool(gdn_checkpoint)
        if gdn_checkpoint:
            self.prefill_runner._gdn_cp_state = self._gdn_cp_state
            self.prefill_runner._gdn_cp_conv = self._gdn_cp_conv
        # 去 rollback：初始状态来源（首步 verify / 正常 prefill 走 pool，INIT_FROM_CP=False）
        self.prefill_runner._gdn_init_from_cp = bool(init_from_cp)
        if init_from_cp:
            self.prefill_runner._gdn_init_state_s = init_state_s
            self.prefill_runner._gdn_init_state_c = init_state_c
        if not graph_mode:
            # host 侧写 buffer[0]：graph 捕获时会被 bake 成 capture 时的值，replay 不重读。
            # graph_mode 下 gdn_slot 由 _verify_model_graph 在 capture 前 eager 写好。
            self.prefill_runner._gdn_prefill_seq_idx[0] = gdn_slot

        try:
            meta = PrefillMeta(
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                position_ids=positions, slot_mapping=slot_mapping,
                block_table=block_table, n_seqs=1,
                max_seqlen_q=max_sq, max_seqlen_k=max_sk)

            # Bug #2 修复：verify（gdn_checkpoint=True）层间保持 (mlp_out, residual) 分离
            # （不预加 bf16），pre-attention norm 用 fused rmsnorm1_residual（mean_sq 在
            # fp32 mlp_out+residual 上算，对齐 decode compute_next_qkv→rmsnorm1_residual）。
            # 原 rmsnorm1(h_bf16) 在 bf16 舍入后 residual 上算 mean_sq → 1-ULP 差经 48 GDN
            # 层×129 步累积 → margin1.75 翻转 → spec target 漂移进循环。prefill（M=P，
            # gdn_checkpoint=False）走旧 prefill（h 完整残差流，rmsnorm1(h_bf16) 对齐 HF，
            # 64-token HF 对齐不受影响）。
            if gdn_checkpoint:
                mlp_out = h  # layer 0：embed 输出
                residual = None
                for layer_idx in range(self.num_layers):
                    block = self.blocks[layer_idx]
                    mlp_out, residual, h = self.adapter.prefill_verify(
                        block, mlp_out, residual, layer_idx, self.prefill_runner,
                        self.cache_manager, meta)
                    if collect_aux_from is not None and layer_idx in self.aux_index:
                        ai = self.aux_index[layer_idx]
                        if graph_mode:
                            # 固定目的地（graph 安全）；replay 后拷到 aux_cache 正确偏移。
                            self._aux_tmp[ai].copy_(h)
                        else:
                            start = collect_aux_from
                            end = start + M
                            if end <= self.max_len:
                                self.aux_cache[ai, start:end].copy_(h)
            else:
                for layer_idx in range(self.num_layers):
                    block = self.blocks[layer_idx]
                    h = self.adapter.prefill(block, h, layer_idx, self.prefill_runner,
                                             self.cache_manager, meta)
                    if collect_aux_from is not None and layer_idx in self.aux_index:
                        ai = self.aux_index[layer_idx]
                        if graph_mode:
                            # 固定目的地（graph 安全）；replay 后拷到 aux_cache 正确偏移。
                            self._aux_tmp[ai].copy_(h)
                        else:
                            start = collect_aux_from
                            end = start + M
                            if end <= self.max_len:
                                self.aux_cache[ai, start:end].copy_(h)

            h = self.final_norm(h)
            if graph_mode:
                # 只到 final_norm（写固定 _vhidden），lm_head 在 replay 后 eager 跑。
                self._vhidden.copy_(h)
                return self._vhidden
            out = self.lm_head(h)
        finally:
            # 复位共享状态：verify GEMM 开关 + GDN 检查点/初始状态开关（engine 正常 prefill/decode 依赖）
            set_verify_gemm(False)
            self.prefill_runner._gdn_cp_enabled = False
            self.prefill_runner._gdn_init_from_cp = False
        return out

    # ------------------------------------------------------------------
    # 阶段薄封装：_forward 的两个调用语义，参数各自定死，调用点自文档
    # ------------------------------------------------------------------
    def _prefill(self, prompt_ids, slot_mapping, gdn_slot):
        """prefill 阶段：整条 prompt 一次 forward。收集 aux[0:P]，无 GDN 检查点
        （prefill 后状态留在 pool，供首步 verify 作初始状态），走反量化 matmul。"""
        P = prompt_ids.shape[0]
        positions = self._arange[:P]
        self._cu_q[0] = 0
        self._cu_q[1] = P
        self._cu_k[0] = 0
        self._cu_k[1] = P
        return self._forward(prompt_ids, positions, slot_mapping[:P],
                             self._cu_q, self._cu_k, self._bt,
                             P, P, gdn_slot, collect_aux_from=0, gdn_checkpoint=False)

    def _verify(self, anchor, draft_tokens, kv_len, slot_mapping, gdn_slot):
        """verify 阶段：anchor + N 个 draft（M=1+N token）一次 forward。
        开 GDN 逐 token 检查点（accept 后按 accepted 回跳），收集 aux[kv_len-1:...]；
        非首步 GDN 初始状态直接读 checkpoint[accepted_prev]（去 rollback）。
        返回 vlogits [M, vocab]。

        稳态（非首步，有检查点）且 verify graph 已捕获 → 走 CUDA graph replay
        （_verify_graph_replay）；首步（init_from_cp=False，GDN 初始状态在 pool）
        或 graph 未捕获 → eager。
        """
        M = 1 + self.N
        # 稳态 verify 走 CUDA graph（固定 M=1+N shape，GDN 初始状态从 checkpoint 读）
        if self._verify_model_graph is not None and self._gdn_accepted_prev is not None:
            return self._verify_graph_replay(anchor, draft_tokens, kv_len,
                                             slot_mapping, gdn_slot)
        # ---- eager 路径（首步 / graph 未捕获）----
        self._verify_ids[0] = anchor
        self._verify_ids[1:] = draft_tokens
        vpos = self._arange[kv_len - 1: kv_len - 1 + M]
        vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
        self._cu_q[0] = 0
        self._cu_q[1] = M
        self._cu_k[0] = 0
        self._cu_k[1] = kv_len - 1 + M
        # 去 rollback：非首步 GDN 初始状态直接读 checkpoint[accepted_prev]。
        # CUDA graph 安全：INIT_STATE 传【完整 checkpoint buffer base】，token 索引
        # 写 device buffer _gdn_init_idx[0]（kernel 内 tl.load 读，非 bake 进指针）。
        accepted_prev = self._gdn_accepted_prev
        if accepted_prev is not None:
            self.prefill_runner._gdn_init_idx[0] = accepted_prev
        return self._forward(self._verify_ids, vpos, vslot, self._cu_q, self._cu_k,
                             self._bt, M, kv_len - 1 + M, gdn_slot,
                             collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                             init_from_cp=accepted_prev is not None,
                             init_state_s=(self._gdn_cp_state
                                           if accepted_prev is not None else None),
                             init_state_c=(self._gdn_cp_conv
                                           if accepted_prev is not None else None))

    # ------------------------------------------------------------------
    # verify CUDA graph：捕获 + 稳态 replay
    # ------------------------------------------------------------------
    def capture_verify_model_graph(self):
        """捕获 verify（M=1+N 固定 shape，稳态 init_from_cp=True）的 CUDA graph。

        固定 buffer（replay 前 eager 填，graph 读 device 内存）：
          - verify_ids / vpos / vslot / cu_k：动态输入（每步变），固定地址 + copy_。
          - gdn_slot（_gdn_prefill_seq_idx[0]）/ init token（_gdn_init_idx[0]）：
            device buffer，graph 内 tl.load 读（非 capture 时 bake 的标量）。
        固定地址原地读写（graph 安全）：
          - GDN 递归/conv 状态池（class 级单例）+ 检查点 buffer（_gdn_cp_state/conv）。
          - paged KV cache（store_kvcache 写 + flash 读，block_table=self._bt 固定）。
        graph 内写固定目的地、replay 后 eager 补：
          - aux 写 _aux_tmp[ai]（固定），replay 后拷到 aux_cache[ai, kv_len-1:...]。
          - final_norm 写 _vhidden（固定），replay 后 eager 跑 lm_head。
        失败（某 kernel 不可捕获）→ 留 _verify_model_graph=None，回退 eager。
        """
        if self._verify_model_graph is not None:
            return
        device = self.device
        M = 1 + self.N
        gdn_slot = self._gdn_alloc()
        self.prefill_runner._gdn_state_pool[gdn_slot] = 0
        self.prefill_runner._gdn_conv_state_pool[gdn_slot] = 0
        seq_id = 999_998
        ok, slot_mapping, _ = self.cache_manager.alloc(seq_id, self.max_len, None)
        if not ok:
            self._gdn_free(gdn_slot)
            raise RuntimeError("verify graph capture: KV cache alloc failed")
        blocks = self.cache_manager._blocks[seq_id]
        self._bt.fill_(-1)
        self._bt[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
        try:
            # 填固定 buffer（dummy 值，capture 只记录结构/指针）
            self._verify_ids.fill_(0)
            kv_len = M + 1  # dummy：让 vpos/vslot/cu_k 落在合法范围
            self._vpos.copy_(self._arange[kv_len - 1: kv_len - 1 + M])
            self._vslot.copy_(slot_mapping[kv_len - 1: kv_len - 1 + M])
            self._cu_q[0] = 0
            self._cu_q[1] = M
            self._cu_k[0] = 0
            self._cu_k[1] = kv_len - 1 + M
            # GDN 稳态：gdn_slot + init token index（device buffer，graph 内读）
            self.prefill_runner._gdn_prefill_seq_idx[0] = gdn_slot
            self.prefill_runner._gdn_init_idx[0] = 0
            # warmup（eager，触发所有层 verify kernel 编译 + 稳定 allocator）
            for _ in range(3):
                self._forward(self._verify_ids, self._vpos, self._vslot,
                              self._cu_q, self._cu_k, self._bt,
                              M, self.max_len, gdn_slot,
                              collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                              init_from_cp=True,
                              init_state_s=self._gdn_cp_state,
                              init_state_c=self._gdn_cp_conv,
                              graph_mode=True)
            torch.cuda.synchronize()
            # capture（max_seqlen_k 传固定上界 self.max_len：flash varlen 的 grid 由
            # seqlen_q=M 定、K-loop 由 cu_seqlen_k device buffer 定，max_seqlen_k 不进
            # grid/loop，故固定上界安全且 graph 可重放）
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                self._forward(self._verify_ids, self._vpos, self._vslot,
                              self._cu_q, self._cu_k, self._bt,
                              M, self.max_len, gdn_slot,
                              collect_aux_from=kv_len - 1, gdn_checkpoint=True,
                              init_from_cp=True,
                              init_state_s=self._gdn_cp_state,
                              init_state_c=self._gdn_cp_conv,
                              graph_mode=True)
            self._verify_model_graph = g
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"verify CUDA graph 捕获失败，回退 eager: {e}")
            self._verify_model_graph = None
        finally:
            self.cache_manager.free(seq_id)
            self._gdn_free(gdn_slot)
            # 复位共享 GDN 标志（capture 后 engine 正常 prefill/decode 依赖）
            self.prefill_runner._gdn_cp_enabled = False
            self.prefill_runner._gdn_init_from_cp = False

    def _verify_graph_replay(self, anchor, draft_tokens, kv_len, slot_mapping,
                             gdn_slot):
        """稳态 verify 走 CUDA graph replay。返回 vlogits [M, vocab]。

        replay 前 eager 填固定 buffer（graph 读 device 内存）；replay 后 eager 补
        aux 拷贝（_aux_tmp→aux_cache 正确偏移）+ lm_head（_vhidden→vlogits）。
        """
        M = 1 + self.N
        self._verify_ids[0] = anchor
        self._verify_ids[1:] = draft_tokens
        self._vpos.copy_(self._arange[kv_len - 1: kv_len - 1 + M])
        self._vslot.copy_(slot_mapping[kv_len - 1: kv_len - 1 + M])
        self._cu_q[0] = 0
        self._cu_q[1] = M
        self._cu_k[0] = 0
        self._cu_k[1] = kv_len - 1 + M
        # GDN：gdn_slot + init token index（device buffer，graph 内 tl.load 读）
        self.prefill_runner._gdn_prefill_seq_idx[0] = gdn_slot
        self.prefill_runner._gdn_init_idx[0] = self._gdn_accepted_prev
        # replay（GDN 状态池/检查点/paged KV 原地读写，aux→_aux_tmp，final_norm→_vhidden）
        self._verify_model_graph.replay()
        # aux：固定 _aux_tmp[ai] → aux_cache[ai, kv_len-1:...]（目的地偏移每步变，eager 拷）
        for ai in range(self.num_aux):
            self.aux_cache[ai, kv_len - 1: kv_len - 1 + M].copy_(self._aux_tmp[ai])
        # lm_head（replay 后 eager 跑，_vhidden 是 final_norm 输出）
        return self.lm_head(self._vhidden)

    # ------------------------------------------------------------------
    # draft CUDA graph：捕获 + 稳态 replay（与 verify graph 分开管理）
    # ------------------------------------------------------------------
    def capture_draft_model_graph(self):
        """捕获 draft 模型 forward（embed → 5 层 decoder → select_draft_tokens）的
        CUDA graph。与 verify graph 分开：draft 是独立小草稿模型（不碰 GDN 状态池/
        paged KV），只读 draft context KV（_ctx_k/_ctx_v）+ 共享 embed/lm_head。

        难点：draft attention 读 context KV [0:ctx_len]，ctx_len 每步增长（变长），
        graph 需固定 shape。解法：固定读 [0:C]（C=max_len），用 attn_mask 屏蔽
        [ctx_len:C) 的无效 context 位置（exp(-inf)=0，softmax 结果与只读 [ctx_len)
        完全一致）。query 位置 / mask / anchor 都是固定 buffer，replay 前 eager 填。

        固定 buffer（replay 前 eager 填，graph 读 device 内存）：
          - _query_ids[0]=anchor / _anchor_t[0]=anchor / _dpos / _dmask：动态输入。
        固定地址原地读（graph 安全）：
          - _ctx_k/_ctx_v（draft context KV 常驻 buffer，[0:C] 固定切片）。
        graph 内写固定目的地：
          - _draft_out（[N] 提议 token）。
        失败（某 kernel 不可捕获）→ 留 _draft_model_graph=None，回退 eager。

        注意（默认关，MICRO_DRAFT_GRAPH=1 才开）：attn_mask 强制 SDPA 走
        mem-efficient 后端，与原变长 flash 后端在 bf16 级有差异，经 draft selector
        的 argmax 放大成完全不同的提议 token → 接受率从 1.98 崩到 0.008（3x 吞吐
        回归）。draft 提议质量直接决定投机解码吞吐，故默认走 eager（与原行为逐
        token 一致）。此 graph 保留作结构拆分 + A/B 对比用。
        """
        if self._draft_model_graph is not None:
            return
        device = self.device
        M = 1 + self.N
        C = self.max_len
        try:
            # 填固定 buffer（dummy 值，capture 只记录结构/指针）
            self._query_ids.fill_(0)
            self._anchor_t[0] = 0
            self._dpos.copy_(self._arange[:M])
            self._dmask.fill_(0)
            # warmup（eager，触发 draft 各层 kernel 编译 + 稳定 allocator）
            for _ in range(3):
                self._draft_forward_graph()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                self._draft_forward_graph()
            self._draft_model_graph = g
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"draft CUDA graph 捕获失败，回退 eager: {e}")
            self._draft_model_graph = None

    def _draft_forward_graph(self):
        """draft forward 的 graph 体内（固定 shape，读固定 buffer）。
        返回 [N] 提议 token（写进 _draft_out）。"""
        M = 1 + self.N
        C = self.max_len
        query_embeds = self.embed(self._query_ids)
        if self.input_embedding_scale != 1.0:
            query_embeds = query_embeds * self.input_embedding_scale
        context_kv = [
            (self._ctx_k[i, :C], self._ctx_v[i, :C])
            for i in range(self.draft.num_layers)
        ]
        out = self.draft.forward(self._query_ids, self._dpos,
                                 input_embeds=query_embeds, context_kv=context_kv,
                                 attn_mask=self._dmask)
        # out: [1+N, hidden]，取 [1:]（N 个 mask 位置）
        if self.draft.candidate_selector is not None:
            draft = self.draft.select_draft_tokens(
                out[1:].unsqueeze(0), self._anchor_t)
        else:
            logits = self.lm_head(out[1:])
            draft = logits.argmax(dim=-1).unsqueeze(0)
        self._draft_out.copy_(draft[0])
        return self._draft_out

    def _draft_graph_replay(self, anchor, ctx_len):
        """稳态 draft 走 CUDA graph replay。返回 [N] 提议 token。

        replay 前 eager 填固定 buffer（graph 读 device 内存）：anchor / query 位置 /
        attn_mask（屏蔽 [ctx_len:C) 无效 context）。replay 后读 _draft_out。"""
        M = 1 + self.N
        C = self.max_len
        self._query_ids[0] = anchor
        self._anchor_t[0] = anchor
        self._dpos.copy_(self._arange[ctx_len: ctx_len + M])
        # attn_mask：[0:ctx_len] 有效(0)，[ctx_len:C) 屏蔽(-inf)，query [C:C+M] 有效(0)
        self._dmask.fill_(0)
        if ctx_len < C:
            self._dmask[:, ctx_len:C].fill_(float("-inf"))
        self._draft_model_graph.replay()
        return self._draft_out

    # ------------------------------------------------------------------
    # GDN 状态回滚
    # ------------------------------------------------------------------
    def _gdn_rollback(self, gdn_slot, accepted):
        """把 GDN 递归/conv 状态回滚到 checkpoint[accepted]。
        checkpoint[t] = 处理完 verify_ids[t]（0-indexed）后的状态。
        保留 anchor + accepted 个 draft = 前 1+accepted 个 token 的状态 = checkpoint[accepted]。"""
        if self._gdn_cp_state is None:
            return
        pool = self._gdn_pool()
        pool["state"][gdn_slot].copy_(self._gdn_cp_state[accepted])
        pool["conv"][gdn_slot].copy_(self._gdn_cp_conv[accepted])

    # ------------------------------------------------------------------
    # draft 提议
    # ------------------------------------------------------------------
    def _draft_propose(self, anchor, kv_len):
        """draft 模型提议 N 个 token。

        anchor: 最后已提交 token（位置 kv_len-1）。context = tokens[0:kv_len-1]。
        返回 [N] int64 提议 token。
        """
        ctx_len = kv_len - 1
        # 稳态 draft 走 CUDA graph（固定 context 长度 C + attn_mask 屏蔽无效位）。
        # 需 ctx_len>0（graph 固定读 [0:C] context，ctx_len=0 无 context 走 eager）。
        if self._draft_model_graph is not None and ctx_len > 0:
            # 增量 context KV 补填（同 eager 路径，保证 [0:ctx_len] 有效）
            done = self._ctx_kv_done
            if done < ctx_len:
                if done > 0:
                    aux = self.aux_cache[:, done:ctx_len].permute(1, 0, 2).reshape(ctx_len - done, -1)
                else:
                    aux = self.aux_cache[:, :ctx_len].permute(1, 0, 2).reshape(ctx_len, -1)
                combined = self.draft.combine_hidden_states(aux)  # [ctx_len-done, hidden]
                self.draft.fill_context_kv(
                    combined, self._arange[done:ctx_len],
                    self._ctx_k, self._ctx_v, done, ctx_len)
                self._ctx_kv_done = ctx_len
            return self._draft_graph_replay(anchor, ctx_len)

        if ctx_len > 0:
            # 增量 context KV：热路径（generate 主循环里）_ctx_kv_done == ctx_len，
            # 这里纯切片读常驻 buffer，零重算。对外部调用方（benchmark 脚本直接拿合成
            # aux 调、不经 generate 流程填 buffer）自愈合：补填 [done, ctx_len)。
            done = self._ctx_kv_done
            if done < ctx_len:
                if done > 0:
                    aux = self.aux_cache[:, done:ctx_len].permute(1, 0, 2).reshape(ctx_len - done, -1)
                else:
                    aux = self.aux_cache[:, :ctx_len].permute(1, 0, 2).reshape(ctx_len, -1)
                combined = self.draft.combine_hidden_states(aux)  # [ctx_len-done, hidden]
                self.draft.fill_context_kv(
                    combined, self._arange[done:ctx_len],
                    self._ctx_k, self._ctx_v, done, ctx_len)
                self._ctx_kv_done = ctx_len
            context_kv = [
                (self._ctx_k[i, :ctx_len], self._ctx_v[i, :ctx_len])
                for i in range(self.draft.num_layers)
            ]
        else:
            context_kv = None

        # query = [anchor] + [mask]*N，位置 [kv_len-1, kv_len+N)。
        # 位置就是连续 arange，直接取常驻 _arange 的零拷贝切片（_pos_buf 是 empty
        # 未初始化，读它拿到垃圾 index → rope gather OOB，故弃用 _pos_buf）。
        self._query_ids[0] = anchor
        query_embeds = self.embed(self._query_ids)
        if self.input_embedding_scale != 1.0:
            query_embeds = query_embeds * self.input_embedding_scale
        query_pos = self._arange[kv_len - 1: kv_len - 1 + 1 + self.N]
        out = self.draft.forward(self._query_ids, query_pos,
                                 input_embeds=query_embeds, context_kv=context_kv)
        # out: [1+N, hidden]，取 [1:]（N 个 mask 位置）
        if self.draft.candidate_selector is not None:
            # DFlash2 完整选 token：selector 边打分 + 贪心 walk
            self._anchor_t[0] = anchor
            draft = self.draft.select_draft_tokens(
                out[1:].unsqueeze(0), self._anchor_t)
            return draft[0]  # [N]
        # 回退：直接 argmax
        logits = self.lm_head(out[1:])
        return logits.argmax(dim=-1)  # [N]

    # ------------------------------------------------------------------
    # 增量 context KV 填充
    # ------------------------------------------------------------------
    def _fill_ctx_kv(self, start, end):
        """把 aux_cache[:, start:end] 投影成 draft context KV 写进常驻 buffer。
        start/end 是绝对位置；要求 aux_cache 在 [start,end) 已写好（prefill/verify 产出）。
        只算 end-start 个位置（增量），不重算 [0,start)。"""
        if end <= start or end > self.max_len:
            return
        aux = self.aux_cache[:, start:end].permute(1, 0, 2).reshape(end - start, -1)
        combined = self.draft.combine_hidden_states(aux)  # [end-start, hidden]
        self.draft.fill_context_kv(
            combined, self._arange[start:end],
            self._ctx_k, self._ctx_v, start, end)
        self._ctx_kv_done = end

    # ------------------------------------------------------------------
    # 预热：编译 verify int8 GEMM kernel（verify M=1+N 的各层 shape）
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def warmup(self):
        """跑一次 dummy verify forward（M=1+N，gdn_checkpoint=True），触发所有层
        的 verify int8 GEMM 编译（TileLang 每 (M,N,K,dtype) 一次 ~3s/shape；
        Triton 首调 JIT）。放 init 时做，避免首个真实 verify 卡在编译
        （否则 e2e 吞吐被一次性编译拉低）。"""
        device = self.device
        M = 1 + self.N
        gdn_slot = self._gdn_alloc()
        self.prefill_runner._gdn_state_pool[gdn_slot] = 0
        self.prefill_runner._gdn_conv_state_pool[gdn_slot] = 0
        # 临时 KV slot（dummy，只为让 _forward 跑通编译）
        seq_id = 999_999
        ok, slot_mapping, _ = self.cache_manager.alloc(seq_id, M, None)
        if ok:
            blocks = self.cache_manager._blocks[seq_id]
            bt = torch.full((1, self.cache_manager.max_seq_blocks), -1,
                            dtype=torch.int32, device=device)
            bt[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
            input_ids = torch.zeros(M, dtype=torch.int64, device=device)
            positions = torch.arange(M, device=device, dtype=torch.int64)
            cu_q = torch.tensor([0, M], device=device, dtype=torch.int32)
            cu_k = torch.tensor([0, M], device=device, dtype=torch.int32)
            try:
                self._forward(input_ids, positions, slot_mapping, cu_q, cu_k,
                              bt, M, M, gdn_slot,
                              collect_aux_from=0, gdn_checkpoint=True)
            finally:
                self.cache_manager.free(seq_id)
        self._gdn_free(gdn_slot)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], max_tokens: int,
                 eos_token_id: Optional[int] = None,
                 on_tokens=None, ignore_eos: bool = False) -> List[int]:
        """投机解码生成。返回新生成的 token 列表。

        on_tokens: 可选回调，每提交一批 token（首 token / 每步 accepted+bonus）时
            调用 on_tokens(List[int])。用于真流式（SSE 逐 token 推送）。None=不回调。
        ignore_eos: True 时遇到 EOS 不停（跑满 max_tokens），对齐 OpenAI/vllm bench 语义。
        """
        # ignore_eos 时直接置 None，下方两处 EOS 检查自动短路
        if ignore_eos:
            eos_token_id = None
        device = self.device
        P = len(prompt_ids)
        # 先把真实 prompt token 载入静态 buffer（_prompt_buf 是 empty 未初始化，
        # 必须先 copy 真实值再取切片，否则 embed_tokens 读到垃圾 index → gather OOB）。
        self._prompt_buf[:P].copy_(torch.tensor(prompt_ids, dtype=torch.int64, device=device))
        prompt_ids = self._prompt_buf[:P]
        if P >= self.max_len:
            raise RuntimeError(f"prompt 过长 ({P} >= {self.max_len})")
        # 留 N 个 slot 给 verify 的 draft token（verify 写 [kv_len-1, kv_len-1+1+N)）
        max_tokens = max(1, min(max_tokens, self.max_len - P - self.N))

        # 重置统计
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0
        if self._pos_stats is not None:
            self._pos_stats = self._new_pos_stats()
        self.aux_cache.zero_()
        self._ctx_kv_done = 0

        # ---- 分配 KV cache（一次性 max_len 个 slot）----
        seq_id = 1_000_000
        ok, slot_mapping, _ = self.cache_manager.alloc(seq_id, self.max_len, None)
        if not ok:
            raise RuntimeError("spec decode: KV cache alloc failed")
        # 静态 block_table（复用 init 预分配的常驻 _bt buffer，避免每次 generate 新建）
        blocks = self.cache_manager._blocks[seq_id]
        bt = self._bt
        bt.fill_(-1)
        bt[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)

        # ---- 分配 GDN slot + 清零状态（新序列从空状态开始）----
        gdn_slot = self._gdn_alloc()
        self.prefill_runner._gdn_state_pool[gdn_slot] = 0
        self.prefill_runner._gdn_conv_state_pool[gdn_slot] = 0

        try:
            # ---- prefill prompt（收集 aux[0:P]）----
            logits = self._prefill(prompt_ids, slot_mapping, gdn_slot)
            anchor = int(logits[-1].argmax())
            generated = [anchor]
            kv_len = P + 1
            self.total_generated = 1
            if on_tokens is not None:
                on_tokens([anchor])
            # 首步 verify 无检查点（prefill 后 GDN 状态在 pool），初始状态从 pool 读。
            self._gdn_accepted_prev = None
            # 增量 context KV：prefill 已写好 aux[0:P]，投影填充 [0,P)（一次性 O(P)，
            # 后续每步只增量填新 accepted 的位置）。
            self._fill_ctx_kv(0, P)

            # ---- 投机解码主循环 ----
            while len(generated) < max_tokens:
                if eos_token_id is not None and anchor == eos_token_id:
                    break
                if kv_len + self.N > self.max_len:
                    break

                # 1. draft 提议
                draft_tokens = self._draft_propose(anchor, kv_len)  # [N]
                # 2. verify（1+N token，开 GDN 检查点，收集 aux；输入走常驻 buffer）
                vlogits = self._verify(anchor, draft_tokens, kv_len, slot_mapping, gdn_slot)
                self.total_steps += 1

                # 3. 贪心接受。draft/target 各 .cpu() 一次（共 2 次同步），Python 侧
                # 纯 int 比较——替代原来逐元素 int() 的 ~N+2 次 CPU-GPU 同步。
                target_preds = vlogits.argmax(dim=-1)  # [M]
                d_cpu = draft_tokens.cpu().tolist()
                t_cpu = target_preds.cpu().tolist()
                accepted = 0
                for i in range(self.N):
                    if d_cpu[i] == t_cpu[i]:
                        accepted += 1
                    else:
                        break
                bonus = t_cpu[accepted]
                # per-position 诊断（MICRO_SPEC_POS_STATS=1）：本步 ctx_len=kv_len-1
                if self._pos_stats is not None:
                    st = self._pos_stats
                    b = self._pos_bucket(kv_len - 1)
                    st["steps"] += 1
                    st["steps_b"][b] += 1
                    for i in range(self.N):
                        if d_cpu[i] == t_cpu[i]:
                            st["match"][b][i] += 1
                        if accepted > i:
                            st["prefix"][b][i] += 1

                # 4. 去 rollback：不再 copy_ 回 pool。记录 accepted，下一步 verify 的
                #    GDN 初始状态直接读 checkpoint[accepted]（见上方 init_from_cp）。
                self._gdn_accepted_prev = accepted

                # 5. 提交 accepted 个 draft + 1 个 bonus
                new_tokens = d_cpu[:accepted] + [bonus]
                generated.extend(new_tokens)
                if on_tokens is not None:
                    on_tokens(new_tokens)
                self.total_accepted += accepted
                self.total_generated += len(new_tokens)
                # 6. 增量 context KV：本步 verify 已写 aux [kv_len-1, kv_len-1+M)，其中
                #    [kv_len-1, kv_len+accepted)（accepted+1 个位置，含 anchor）是下一步
                #    draft 的新 context（下一步 ctx_len = kv_len+accepted+1-1 =
                #    kv_len+accepted）。只投影这 accepted+1 个位置写进常驻 buffer，不重算
                #    [0, kv_len-1)——把旧的每步 O(ctx_len) 全量重算降成 O(accepted+1)≈O(1)，
                #    整段 O(n²)→O(n)。_ctx_kv_done 追到 kv_len+accepted = 下一步 ctx_len。
                self._fill_ctx_kv(kv_len - 1, kv_len + accepted)
                kv_len += len(new_tokens)
                anchor = bonus

                if eos_token_id is not None and anchor == eos_token_id:
                    break

            return generated[:max_tokens]
        finally:
            self.cache_manager.free(seq_id)
            self._gdn_free(gdn_slot)

    @property
    def avg_acceptance(self) -> float:
        return self.total_accepted / self.total_steps if self.total_steps else 0.0
