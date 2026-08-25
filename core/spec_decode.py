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
from typing import List, Optional

import torch

from models.base import PrefillMeta
from kernel.gemv_int8 import set_force_gemm


class SpecDecodeController:
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

        # GDN 状态检查点 buffer（verify 时逐 token 存，接受后回滚）
        self._gdn_cp_state = None
        self._gdn_cp_conv = None
        self._alloc_gdn_checkpoint()

        # 统计
        self.total_accepted = 0
        self.total_steps = 0
        self.total_generated = 0

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
    def _target_forward(self, input_ids, positions, slot_mapping,
                        cu_q, cu_k, block_table, max_sq, max_sk,
                        gdn_slot, collect_aux_from, gdn_checkpoint):
        """跑一次 target prefill forward，返回 logits [M, vocab]。

        collect_aux_from: 从哪个绝对位置开始收集 aux（None=不收集），收集 M 个。
        gdn_checkpoint: 是否开 GDN 逐 token 检查点（verify 用）。
        """
        M = input_ids.shape[0]
        h = self.embed(input_ids)

        # verify（M=1+N≈8）强制 TileLang int8 分块 GEMM：int8 GEMV 对 M>1 把权重读
        # M 次（27GB×8=216GB），GEMM 权重 HBM 只读一次（shared 内 dequant），快 12-31x。
        # prefill（M 大）本就走反量化 matmul，不受影响。
        set_force_gemm(bool(gdn_checkpoint))

        # GDN 检查点开关（graph=prefill_runner 的 buffer）。prefill_runner 与 engine
        # 正常 prefill 路径【共享】，故必须在 finally 里复位 _gdn_cp_enabled=False，
        # 否则后续非 spec prefill（M 大）会往 8-token 检查点 buffer 写 M 个 → 越界。
        self.prefill_runner._gdn_cp_enabled = bool(gdn_checkpoint)
        if gdn_checkpoint:
            self.prefill_runner._gdn_cp_state = self._gdn_cp_state
            self.prefill_runner._gdn_cp_conv = self._gdn_cp_conv
        self.prefill_runner._gdn_prefill_seq_idx[0] = gdn_slot

        try:
            meta = PrefillMeta(
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                position_ids=positions, slot_mapping=slot_mapping,
                block_table=block_table, n_seqs=1,
                max_seqlen_q=max_sq, max_seqlen_k=max_sk)

            for layer_idx in range(self.num_layers):
                block = self.blocks[layer_idx]
                h = self.adapter.prefill(block, h, layer_idx, self.prefill_runner,
                                         self.cache_manager, meta)
                if collect_aux_from is not None and layer_idx in self.aux_index:
                    ai = self.aux_index[layer_idx]
                    start = collect_aux_from
                    end = start + M
                    if end <= self.max_len:
                        self.aux_cache[ai, start:end].copy_(h)

            h = self.final_norm(h)
            out = self.lm_head(h)
        finally:
            # 复位共享状态：force_gemm + GDN 检查点开关（engine 正常 prefill/decode 依赖）
            set_force_gemm(False)
            self.prefill_runner._gdn_cp_enabled = False
        return out

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
        if ctx_len > 0:
            # aux: [num_aux, ctx_len, hidden] → [ctx_len, num_aux*hidden]
            aux = self.aux_cache[:, :ctx_len].permute(1, 0, 2).reshape(ctx_len, -1)
            combined = self.draft.combine_hidden_states(aux)  # [ctx_len, hidden]
            ctx_pos = self._arange[:ctx_len]
            context_kv = self.draft.precompute_context_kv(combined, ctx_pos)
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
    # 预热：编译 TileLang int8 GEMM kernel（verify M=1+N 的各层 shape）
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def warmup(self):
        """跑一次 dummy verify forward（M=1+N，gdn_checkpoint=True），触发所有层
        的 TileLang int8 GEMM 编译（每 (M,N,K,dtype) 一次，~3s/shape）。放 init 时
        做，避免首个真实 verify 卡在编译（否则 e2e 吞吐被一次性编译拉低）。"""
        device = self.device
        M = 1 + self.N
        gdn_slot = self._gdn_alloc()
        self.prefill_runner._gdn_state_pool[gdn_slot] = 0
        self.prefill_runner._gdn_conv_state_pool[gdn_slot] = 0
        # 临时 KV slot（dummy，只为让 _target_forward 跑通编译）
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
                self._target_forward(input_ids, positions, slot_mapping, cu_q, cu_k,
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
                 eos_token_id: Optional[int] = None) -> List[int]:
        """投机解码生成。返回新生成的 token 列表。"""
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
        self.aux_cache.zero_()

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
            # 位置/cu_seqlens 走常驻 buffer（零运行期分配）：positions=arange 前缀切片，
            # cu_q/cu_k 前缀写 [0, P]。
            positions = self._arange[:P]
            self._cu_q[0] = 0
            self._cu_q[1] = P
            self._cu_k[0] = 0
            self._cu_k[1] = P
            logits = self._target_forward(
                prompt_ids, positions, slot_mapping[:P], self._cu_q, self._cu_k, bt,
                P, P, gdn_slot, collect_aux_from=0, gdn_checkpoint=False)
            anchor = int(logits[-1].argmax())
            generated = [anchor]
            kv_len = P + 1
            self.total_generated = 1

            # ---- 投机解码主循环 ----
            while len(generated) < max_tokens:
                if eos_token_id is not None and anchor == eos_token_id:
                    break
                if kv_len + self.N > self.max_len:
                    break

                # 1. draft 提议
                draft_tokens = self._draft_propose(anchor, kv_len)  # [N]
                # 2. verify（1+N token，开 GDN 检查点，收集 aux[kv_len-1:...]）
                # 输入全部走常驻 buffer（零运行期分配）：verify_ids 前缀写、vpos=arange
                # 切片、vslot=slot_mapping 切片、cu_q/cu_k 前缀写。
                M = 1 + self.N
                self._verify_ids[0] = anchor
                self._verify_ids[1:] = draft_tokens
                vpos = self._arange[kv_len - 1: kv_len - 1 + M]
                vslot = slot_mapping[kv_len - 1: kv_len - 1 + M]
                self._cu_q[0] = 0
                self._cu_q[1] = M
                self._cu_k[0] = 0
                self._cu_k[1] = kv_len - 1 + M
                vlogits = self._target_forward(
                    self._verify_ids, vpos, vslot, self._cu_q, self._cu_k, bt,
                    M, kv_len - 1 + M, gdn_slot,
                    collect_aux_from=kv_len - 1, gdn_checkpoint=True)
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

                # 4. GDN 状态回滚到 checkpoint[accepted]
                self._gdn_rollback(gdn_slot, accepted)

                # 5. 提交 accepted 个 draft + 1 个 bonus
                new_tokens = d_cpu[:accepted] + [bonus]
                generated.extend(new_tokens)
                self.total_accepted += accepted
                self.total_generated += len(new_tokens)
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
