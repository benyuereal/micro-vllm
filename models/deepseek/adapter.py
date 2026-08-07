"""
DeepSeekAdapter - DeepSeek-V2-Lite (MLA + MoE) 适配器。

🔑 架构要点：
    - MLA 注意力：KV cache 存【压缩 latent】(kv_lora_rank 512 + qk_rope 64 = 576)，
      非 per-head 展开。attention 时 kv_b_proj 展开成 per-head k_nope/v。
    - RoPE 只作用 q_pe/k_pe（qk_rope_head_dim=64），需外部旋转后再进 flash_attn_func。
    - softmax_scale = q_head_dim**-0.5 * mscale^2（yarn mscale_all_dim）。
    - MoE：layer 0 dense，layer 1~26 routed(64 experts top6) + shared(2)。

🧩 cache 形状：KVCacheManager 存 [n_blocks, block_size, 1, 576]。
"""
import math
import torch
import torch.nn.functional as F

from models.base import ModelAdapter
from kernel.rmsnorm import rmsnorm, rmsnorm_, rmsnorm_residual_gemm as rmsnorm_residual
from .moe import moe_forward

try:
    from flash_attn import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_with_kvcache = flash_attn_func = flash_attn_varlen_func = None


def _yarn_mscale(scale, mscale):
    """与 HF modeling_deepseek.yarn_get_mscale 完全一致：0.1*mscale*log(scale) + 1.0。"""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepSeekAdapter(ModelAdapter):
    model_type = "deepseek"

    def __init__(self):
        self._latent_dim = None       # kv_lora_rank + qk_rope_head_dim
        self._kv_lora_rank = None
        self._qk_rope = None
        self._qk_nope = None
        self._v_head = None
        self._q_head = None
        self._num_heads = None

    # -------------------- 元信息 --------------------
    def _cfg(self, cfg):
        self._cfg_cache = cfg
        self._kv_lora_rank = cfg.kv_lora_rank               # 512
        self._qk_rope = cfg.qk_rope_head_dim                # 64
        self._qk_nope = cfg.qk_nope_head_dim                # 128
        self._v_head = cfg.v_head_dim                       # 128
        self._q_head = self._qk_nope + self._qk_rope        # 192
        self._num_heads = cfg.num_attention_heads           # 16
        self._latent_dim = self._kv_lora_rank + self._qk_rope  # 576
        self._hidden = cfg.hidden_size
        # MoE 参数
        self._n_experts = cfg.n_routed_experts
        self._top_k = cfg.num_experts_per_tok
        self._moe_inter = cfg.moe_intermediate_size
        self._n_shared = getattr(cfg, "n_shared_experts", None)
        self._first_k_dense = cfg.first_k_dense_replace
        self._scoring_func = getattr(cfg, "scoring_func", "softmax")
        self._norm_topk = getattr(cfg, "norm_topk_prob", False)
        self._routed_scale = getattr(cfg, "routed_scaling_factor", 1.0)
        self._q_lora_rank = getattr(cfg, "q_lora_rank", None)

    def cache_dims(self, cfg):
        self._cfg(cfg)
        # MLA: 单 latent head，cache 存 576 维
        return self._num_heads, 1, self._latent_dim

    def intermediate_size(self, cfg, world_size):
        # DeepSeek 用 MoE，dense 层的 intermediate_size；MoE expert 尺寸单独处理
        return getattr(cfg, "intermediate_size", cfg.moe_intermediate_size) // world_size

    def num_layers(self, cfg):
        return cfg.num_hidden_layers

    def rope_dim(self, cfg):
        self._cfg(cfg)
        return self._qk_rope

    def softmax_scale(self, cfg):
        """与 transformers>=4.56 native DeepseekV2 的 attention_scaling 约定对齐。

        native（modeling_rope_utils）:
            scaling        = qk_head_dim ** -0.5           # 纯
            attention_factor = get_mscale(factor, mscale)
                              / get_mscale(factor, mscale_all_dim)   # mscale 与 mscale_all_dim 都设
                            = get_mscale(factor)            # 仅 mscale_all_dim（兼容旧公式）
            有效 softmax scale = scaling * attention_factor
            cos/sin 再乘 attention_factor（此处等价为直接乘进 scale，结果相同）。

        DeepSeek-V2-Lite 的 mscale==mscale_all_dim==0.707 → attention_factor=1.0
        → 有效 scale = qk_head**-0.5 = 0.0722。
        旧实现误乘 mscale_all_dim^2（=1.59），导致 softmax 过锐、多步生成出乱码。
        """
        self._cfg(cfg)
        scaling = self._q_head ** -0.5
        rope_scaling = getattr(cfg, "rope_scaling", None)
        if rope_scaling is not None:
            mscale = rope_scaling.get("mscale", 0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            factor = rope_scaling.get("factor", 1)
            if mscale and mscale_all_dim:
                attention_factor = _yarn_mscale(factor, mscale) / _yarn_mscale(factor, mscale_all_dim)
            elif mscale_all_dim:
                attention_factor = _yarn_mscale(factor, mscale_all_dim)
            else:
                attention_factor = 1.0
            scaling = scaling * attention_factor
        return scaling

    # -------------------- 权重预处理 --------------------
    def prepare_weights(self, model, world_size, rank):
        blocks = self.blocks(model)
        if getattr(blocks[0].self_attn, "_prepared", False):
            return
        cfg = model.config
        self._cfg(cfg)
        hidden = self._hidden

        for li, block in enumerate(blocks):
            attn = block.self_attn
            # QKV 路径权重（保留为矩阵，不融合——MLA 结构与 GQA 不同）
            # q_proj: [num_heads*q_head, hidden]
            attn._q_w = attn.q_proj.weight.data.clone()
            attn._q_b = attn.q_proj.bias.data.clone() if attn.q_proj.bias is not None else None
            # kv_a_proj_with_mqa: [kv_lora+qk_rope, hidden]
            attn._kva_w = attn.kv_a_proj_with_mqa.weight.data.clone()
            attn._kva_b = attn.kv_a_proj_with_mqa.bias.data.clone() if attn.kv_a_proj_with_mqa.bias is not None else None
            # kv_a_layernorm weight: [kv_lora_rank]
            attn._kva_ln_w = attn.kv_a_layernorm.weight.data.clone()
            attn._kva_ln_eps = attn.kv_a_layernorm.variance_epsilon
            # kv_b_proj: [num_heads*(qk_nope+v_head), kv_lora_rank]
            attn._kvb_w = attn.kv_b_proj.weight.data.clone()
            # o_proj: [hidden, num_heads*v_head]
            attn._o_w = attn.o_proj.weight.data.clone()
            attn._o_b = attn.o_proj.bias.data.clone() if attn.o_proj.bias is not None else None

            # RMSNorm 权重
            block._in_ln_w = block.input_layernorm.weight.data.clone()
            block._in_ln_eps = block.input_layernorm.variance_epsilon
            block._post_ln_w = block.post_attention_layernorm.weight.data.clone()
            block._post_ln_eps = block.post_attention_layernorm.variance_epsilon

            # FFN 权重
            mlp = block.mlp
            if li < self._first_k_dense:
                # dense MLP (SwiGLU: gate_proj + up_proj + down_proj)
                gu = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], dim=0).contiguous()
                mlp._dense_gu = gu.t().contiguous()        # [2*inter, hidden] → [hidden, 2*inter]
                mlp._dense_d = mlp.down_proj.weight.data.t().contiguous()  # [inter, hidden] → [hidden, inter]
                mlp._is_moe = False
            else:
                # MoE: gate weight + 堆叠 experts + shared experts
                gate = mlp.gate
                mlp._gate_w = gate.weight.data.clone()    # [n_experts, hidden]
                experts = mlp.experts
                E = self._n_experts
                inter = self._moe_inter
                hidden = self._hidden
                # 堆叠: _e_gu [E, 2*inter, hidden], _e_d [E, hidden, inter]
                _dev = experts[0].gate_proj.weight.device
                _dt = experts[0].gate_proj.weight.dtype
                e_gu = torch.empty(E, 2 * inter, hidden, dtype=_dt, device=_dev)
                e_d = torch.empty(E, hidden, inter, dtype=_dt, device=_dev)
                for ei, exp in enumerate(experts):
                    e_gu[ei] = torch.cat([exp.gate_proj.weight.data, exp.up_proj.weight.data], dim=0)
                    e_d[ei] = exp.down_proj.weight.data
                mlp._e_gu = e_gu.contiguous()
                mlp._e_d = e_d.contiguous()
                # shared experts（合并成单个大 MLP）
                if self._n_shared is not None and hasattr(mlp, "shared_experts"):
                    se = mlp.shared_experts
                    s_inter = inter * self._n_shared
                    mlp._shared_gu = torch.cat([se.gate_proj.weight.data, se.up_proj.weight.data], dim=0).t().contiguous()
                    mlp._shared_d = se.down_proj.weight.data.t().contiguous()
                else:
                    mlp._shared_gu = mlp._shared_d = None
                mlp._is_moe = True
                # 释放原始 experts 模块
                mlp.experts = None
                mlp.gate = None
                if hasattr(mlp, "shared_experts"):
                    mlp.shared_experts = None

            # 释放原始 attn Linear
            attn.q_proj = None
            attn.kv_a_proj_with_mqa = None
            attn.kv_a_layernorm = None
            attn.kv_b_proj = None
            attn.o_proj = None
            if li < self._first_k_dense:
                block.mlp.gate_proj = block.mlp.up_proj = block.mlp.down_proj = None
            attn._prepared = True
            torch.cuda.empty_cache()

    # -------------------- latent cache 写入（PyTorch 直写，绕过 power-of-2 Triton kernel）--------------------
    # MLA latent 维度 = 576（kv_lora 512 + qk_rope 64）非 2 的幂，框架 store_kvcache 的
    # Triton kernel 要求 head_size 为 2 的幂（tl.arange）。这里用 PyTorch scatter 直写，
    # decode（bs 个 token）与 prefill（B*S 个 token）开销都很小。
    @staticmethod
    def _store_latent(latent_flat, k_cache, v_cache, slots, block_size):
        """latent_flat: [N, 1, latent]，slots: [N] int。写入 k_cache/v_cache 同一 latent。"""
        slots = slots.long()
        block_id = slots // block_size
        offset = slots % block_size
        # k_cache[block_id, offset, 0] = latent_flat[:, 0]
        lv = latent_flat[:, 0, :]                       # [N, latent]
        k_cache[block_id, offset, 0] = lv
        v_cache[block_id, offset, 0] = lv

    @staticmethod
    def _store_latent_batch(latent_b, k_cache, v_cache, slots_b, block_size):
        """latent_b: [bs, 1, 1, latent]，slots_b: [bs, 1] int。每 seq 写 1 个 token。"""
        bs = latent_b.shape[0]
        slots = slots_b.reshape(bs).long()
        block_id = slots // block_size
        offset = slots % block_size
        lv = latent_b[:, 0, 0, :]                       # [bs, latent]
        k_cache[block_id, offset, 0] = lv
        v_cache[block_id, offset, 0] = lv

    # -------------------- 模块访问 --------------------
    def embed(self, model):
        return model.model.embed_tokens

    def blocks(self, model):
        return model.model.layers

    def final_norm(self, model):
        return model.model.norm

    def lm_head(self, model):
        return model.lm_head

    # -------------------- RoPE（仅 qk_rope 维，interleaved 约定）--------------------
    # DeepSeek 用 Llama 风格 interleaved RoPE + YaRN 频率缩放：
    #   cos/sin 全宽 [qk_rope]（cat(freqs,freqs)），旋转用 view(d//2,2).transpose + rotate_half。
    #   inv_freq 用 YaRN 的 extra/inter 混合（与 HF DeepseekV2YarnRotaryEmbedding 完全一致）。
    #   cos/sin 不乘 mscale（HF 中 mscale==mscale_all_dim → _mscale=1）。
    #   与框架 Qwen 的 half-split RoPE 不同，故 DeepSeek 自建全宽 cos/sin pool。
    @staticmethod
    def _yarn_find_correction_dim(num_rotations, dim, base, max_pos):
        return dim * math.log(max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    @classmethod
    def _yarn_inv_freq(cls, dim, base, scaling_factor, original_max_pos, beta_fast, beta_slow, device):
        """复刻 HF DeepseekV2YarnRotaryEmbedding._set_cos_sin_cache 的 inv_freq。"""
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (scaling_factor * base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        low = max(math.floor(cls._yarn_find_correction_dim(beta_fast, dim, base, original_max_pos)), 0)
        high = min(math.ceil(cls._yarn_find_correction_dim(beta_slow, dim, base, original_max_pos)), dim // 2 - 1)
        # linear ramp mask over dim//2 entries
        ar = torch.arange(dim // 2, device=device, dtype=torch.float32)
        if low == high:
            high += 0.001
        ramp = torch.clamp((ar - low) / (high - low), 0.0, 1.0)
        inv_freq_mask = 1.0 - ramp
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        return inv_freq

    def _rope_pool(self, graph, device):
        if getattr(self, "_cos_full", None) is None or self._cos_full.device != device:
            dim = self._qk_rope
            cfg = self._cfg_cache
            base = getattr(cfg, "rope_theta", 10000)
            scaling = getattr(cfg, "rope_scaling", None) or {}
            scaling_factor = scaling.get("factor", 1.0)
            original_max_pos = scaling.get("original_max_position_embeddings", 4096)
            beta_fast = scaling.get("beta_fast", 32)
            beta_slow = scaling.get("beta_slow", 1)
            max_pos = graph.attention.rotary_emb.cos_cache.shape[2]
            inv_freq = self._yarn_inv_freq(dim, base, scaling_factor, original_max_pos,
                                           beta_fast, beta_slow, device)
            t = torch.arange(max_pos, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)        # [max_pos, dim]
            self._cos_full = emb.cos().to(graph.dtype).contiguous()
            self._sin_full = emb.sin().to(graph.dtype).contiguous()
        return self._cos_full, self._sin_full

    def _apply_rope(self, x_pe, cos, sin):
        """interleaved RoPE（与 DeepSeek HF apply_rotary_pos_emb 完全一致）。
        x_pe: [..., qk_rope]，cos/sin: [..., qk_rope]（全宽，cat(freqs,freqs)）。
        HF 先 view(d//2,2).transpose 再 rotate_half，此处等价实现。"""
        # x_pe 末维拆成 (..., d//2, 2) → transpose → reshape，做 deinterleave
        *lead, d = x_pe.shape
        x = x_pe.reshape(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)
        rotate_half = torch.cat((-x[..., d // 2:], x[..., : d // 2]), dim=-1)
        return x * cos + rotate_half * sin

    # -------------------- decode 单层钩子 --------------------
    def compute_qkv(self, block, h, graph, bs):
        # input_layernorm
        rmsnorm_(h, block._in_ln_w, graph._h_buf[:bs], block._in_ln_eps)
        x = graph._h_buf[:bs]
        attn = block.self_attn
        # q: [bs, num_heads*q_head]
        q = F.linear(x, attn._q_w, attn._q_b)
        # kv_a: [bs, kv_lora+qk_rope]
        kva = F.linear(x, attn._kva_w, attn._kva_b)
        # 拆 latent: compressed_kv [bs, kv_lora] | k_pe [bs, qk_rope]
        compressed_kv, k_pe = kva.split([self._kv_lora_rank, self._qk_rope], dim=-1)
        # 缓存到 block 临时槽（attention 钩子用）
        attn._q_cache = q.view(bs, self._num_heads, self._q_head)
        attn._compressed_kv = compressed_kv          # [bs, kv_lora]
        attn._k_pe = k_pe                            # [bs, qk_rope]
        return x  # 返回 normed（attention 钩子需要 residual 由调用方传）

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        rmsnorm_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block_next._in_ln_eps
        )
        x = graph._h_buf[:bs]
        attn = block_next.self_attn
        q = F.linear(x, attn._q_w, attn._q_b)
        kva = F.linear(x, attn._kva_w, attn._kva_b)
        compressed_kv, k_pe = kva.split([self._kv_lora_rank, self._qk_rope], dim=-1)
        attn._q_cache = q.view(bs, self._num_heads, self._q_head)
        attn._compressed_kv = compressed_kv
        attn._k_pe = k_pe
        return x, graph._residual[:bs]

    def attention(self, x_normed, block, layer_idx, bs, graph, cache_manager, block_table):
        """decode MLA attention。

        契约：forward 时 cache_seqlens[i] = 当前序列“下一步”的预期长度（框架在 prefill 后把
        current_position 置为 S+1，故首步 decode seqlens=S+1）。新 token 的逻辑位置 =
        cache_seqlens - 1（0-indexed），写入该 slot，RoPE 用该位置，attention 覆盖
        [0, cache_seqlens-1]（共 cache_seqlens 个 token，全部有效，无空洞）。forward 后 commit +1。
        """
        attn = block.self_attn
        q = attn._q_cache                                  # [bs, H, q_head]
        compressed_kv_new = attn._compressed_kv            # [bs, kv_lora]
        k_pe_new = attn._k_pe                              # [bs, qk_rope]

        k_cache, v_cache = cache_manager.get(layer_idx)    # [n_blocks, block_size, 1, 576]
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]  # [bs]
        new_pos = (cache_lens - 1).long().clamp(min=0)     # 新 token 逻辑位置（0-indexed）
        # 防御：极少数竞态下首步 decode seqlens 可能为 0 → new_pos=-1 → gather 负索引崩溃。
        # 钳到 0 保证不崩（输出可能不准，但避免 device-side assert 拖垮整个 server）。

        # (1) 写入新 token 的 latent [compressed_kv | k_pe] 到位置 new_pos
        latent_new = torch.cat([compressed_kv_new, k_pe_new], dim=-1)  # [bs, 576]
        latent_new = latent_new.view(bs, 1, 1, self._latent_dim)       # [bs, 1, 1head, 576]
        # slot = block_table[seq, new_pos//block_size] * block_size + new_pos%block_size
        # （new_pos 是逻辑位置，必须经 block_table 换算成物理 slot，跨 block 的 seq 才正确）
        slots = self._decode_slots(block_table, new_pos, bs, cache_manager.block_size)
        self._store_latent_batch(latent_new, k_cache, v_cache, slots, cache_manager.block_size)

        # (2) 向量化 gather 每 seq 的全部 latent [0, new_pos]（共 cache_seqlens 个，含新 token）。
        # 构造 [bs, max_len] 物理 slot 矩阵后一次 advanced-index 取 latent。
        # graph 路径：max_len = graph._ds_graph_maxlen（固定桶，消除 .item() 同步），越界 key
        #   经 flash_attn_varlen_func 的 cu_seqlens_k 截断，不参与 attention。
        # eager 路径（序列超桶或未捕获 batch）：max_len = 运行时 total_lens.max()。
        total_lens = cache_lens.long()                     # = new_pos + 1
        use_graph = getattr(graph, "_use_graph_bucket", False)
        graph_maxlen = getattr(graph, "_ds_graph_maxlen", None)
        if use_graph and graph_maxlen is not None:
            max_len = graph_maxlen
        else:
            max_len = int(total_lens.max().item())
        block_size = cache_manager.block_size
        bt = block_table[:bs].long()                       # [bs, max_seq_blocks]
        t_idx = torch.arange(max_len, device=bt.device)    # [max_len]
        blk_idx = t_idx // block_size                      # [max_len]
        off_idx = t_idx % block_size                       # [max_len]
        blk_id = bt[:, blk_idx]                            # [bs, max_len]
        # graph 桶固定 max_len（如 1024），但序列实际只占 ceil(L/block_size) 个 block，
        # block_table 越界列为 -1（非法 block_id）→ 负 slot → k_flat 索引越界崩溃。
        # 将非法 blk_id 钳到 0（指向 block 0 的安全内存），越界位置随后由
        # flash_attn_varlen_func 的 cu_seqlens_k 截断，不参与 attention。
        n_slots = k_cache.shape[0] * block_size
        blk_id = blk_id.clamp(min=0)
        slots = blk_id * block_size + off_idx             # [bs, max_len]
        slots = slots.clamp(min=0, max=n_slots - 1)
        k_flat = k_cache.reshape(-1, self._latent_dim)     # [n_blocks*block_size, 576]
        latents = k_flat[slots.reshape(-1)].view(bs, max_len, self._latent_dim)

        # (3) 展开：compressed_kv → layernorm → kv_b_proj → per-head k_nope, v
        compressed_kv, k_pe_all = latents.split([self._kv_lora_rank, self._qk_rope], dim=-1)
        ckv = rmsnorm(compressed_kv.reshape(-1, self._kv_lora_rank), attn._kva_ln_w, attn._kva_ln_eps)
        kv = F.linear(ckv, attn._kvb_w).view(bs, max_len, self._num_heads, self._qk_nope + self._v_head)
        k_nope, v = kv.split([self._qk_nope, self._v_head], dim=-1)  # [bs, max_len, H, 128]

        # (4) RoPE 仅作用于 q_pe / k_pe (qk_rope 维, interleaved 全宽 cos/sin)
        cos, sin = self._rope_pool(graph, k_cache.device)  # [max_pos, qk_rope] 全宽
        q_nope, q_pe = q.split([self._qk_nope, self._qk_rope], dim=-1)  # [bs, H, 128], [bs, H, 64]
        cos_q = cos[new_pos].unsqueeze(1)                  # [bs, 1, qk_rope]
        sin_q = sin[new_pos].unsqueeze(1)
        q_pe = self._apply_rope(q_pe, cos_q, sin_q)       # [bs, H, 64]
        k_pos = torch.arange(max_len, device=k_pe_all.device).unsqueeze(0)  # [1, max_len]
        cos_k = cos[k_pos]                                # [1, max_len, qk_rope]
        sin_k = sin[k_pos]
        k_pe_rot = self._apply_rope(k_pe_all, cos_k, sin_k)  # [bs, max_len, 64]

        # (5) 拼接 q=[q_nope|q_pe] [bs,H,192]；k=[k_nope|k_pe] [bs, max_len, H, 192]
        q_full = torch.cat([q_nope, q_pe], dim=-1)        # [bs, H, 192]
        k_full = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, self._num_heads, -1)], dim=-1)
        v_fa = torch.nn.functional.pad(v, (0, self._q_head - self._v_head))  # [bs, max_len, H, 192]

        # (6) attention。graph 路径用 flash_attn_varlen_func + cu_seqlens_k 截断每 seq 有效长度
        # （越界 key 不参与，且 cu_seqlens 是 GPU tensor，graph-friendly）；eager 路径用 flash_attn_func。
        if use_graph and graph_maxlen is not None:
            cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=q_full.device)
            cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=q_full.device)
            cu_k[1:] = torch.cumsum(total_lens.to(torch.int32), dim=0)
            # varlen: q [bs,H,D], k/v [bs*max_len,H,D] 按 cu_k 截断每 seq 的 [0,L_i)
            k_v = k_full.reshape(bs * max_len, k_full.shape[-2], k_full.shape[-1])
            v_v = v_fa.reshape(bs * max_len, v_fa.shape[-2], v_fa.shape[-1])
            attn_out = flash_attn_varlen_func(
                q_full, k_v, v_v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=1, max_seqlen_k=max_len,
                softmax_scale=graph._ds_softmax_scale, causal=False)
            attn_out = attn_out[..., :self._v_head].reshape(bs, self._num_heads * self._v_head)
        else:
            q_fa = q_full.unsqueeze(1)                    # [bs, 1, H, 192]
            attn_out = flash_attn_func(q_fa, k_full, v_fa,
                                       softmax_scale=graph._ds_softmax_scale, causal=False)
            attn_out = attn_out[..., :self._v_head].reshape(bs, self._num_heads * self._v_head)

        # (7) o_proj
        return F.linear(attn_out, attn._o_w, attn._o_b)

    def compute_ffn(self, block, attn_out, residual, graph, bs, fast_mode):
        rmsnorm_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        x = graph._h_buf[:bs]
        mlp = block.mlp
        if mlp._is_moe:
            mlp_out = moe_forward(
                x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                self._top_k, self._n_experts,
                mlp._shared_gu, mlp._shared_d, decode=True,
            )
        else:
            # dense SwiGLU（DeepSeek 标准: silu(gate)*up；_dense_gu=cat([gate,up]).t()）
            gate_up = x @ mlp._dense_gu
            gate, up = gate_up.chunk(2, dim=-1)
            mlp_out = (F.silu(gate) * up) @ mlp._dense_d
        return mlp_out, graph._residual[:bs]

    # -------------------- prefill 单层钩子 --------------------
    def prefill_layer(self, block, h, layer_idx, B, S, graph, cache_manager, block_table):
        # input_layernorm
        x = rmsnorm(h, block._in_ln_w, block._in_ln_eps)
        attn = block.self_attn
        q = F.linear(x, attn._q_w, attn._q_b).view(B, S, self._num_heads, self._q_head)
        kva = F.linear(x, attn._kva_w, attn._kva_b)
        compressed_kv, k_pe = kva.split([self._kv_lora_rank, self._qk_rope], dim=-1)
        # 写入 paged cache: latent [B, S, 576]
        latent = torch.cat([compressed_kv, k_pe], dim=-1).view(B, S, 1, self._latent_dim)
        # 构造 slot_mapping: 每个 token 在其 seq 的连续位置 0..S-1
        # block_table 已建好，slot = block_id * block_size + offset
        slots = self._build_prefill_slots(cache_manager, block_table, B, S)
        self._store_latent(latent.reshape(B * S, 1, self._latent_dim),
                           cache_manager.k_caches[layer_idx], cache_manager.v_caches[layer_idx],
                           slots, cache_manager.block_size)

        # attention: 展开 + RoPE + flash_attn_func (varlen-like: 每 seq 长度 S)
        q_nope, q_pe = q.split([self._qk_nope, self._qk_rope], dim=-1)
        cos_full, sin_full = self._rope_pool(graph, compressed_kv.device)
        cos = cos_full[:S]                                # [S, qk_rope] 全宽
        sin = sin_full[:S]
        q_pe = self._apply_rope(q_pe, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))
        # k_pe: [B, S, qr] (3D，单 head) → cos 广播为 [1, S, qr]
        k_pe_rot = self._apply_rope(k_pe, cos.unsqueeze(0), sin.unsqueeze(0))
        # kv_b_proj 展开
        ckv = rmsnorm(compressed_kv.reshape(-1, self._kv_lora_rank), attn._kva_ln_w, attn._kva_ln_eps)
        kv = F.linear(ckv, attn._kvb_w).view(B, S, self._num_heads, self._qk_nope + self._v_head)
        k_nope, v = kv.split([self._qk_nope, self._v_head], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)       # [B, S, H, 192]
        k_full = torch.cat([k_nope, k_pe_rot.unsqueeze(2).expand(-1, -1, self._num_heads, -1)], dim=-1)
        # flash: [B, S, H, D]
        v_pad = torch.nn.functional.pad(v, (0, self._q_head - self._v_head))
        scale = graph._ds_softmax_scale
        attn_out = flash_attn_func(q_full, k_full, v_pad, softmax_scale=scale, causal=True)
        attn_out = attn_out[..., :self._v_head].reshape(B, S, self._num_heads * self._v_head)
        out = F.linear(attn_out, attn._o_w, attn._o_b)

        # FFN
        res = h
        h2 = rmsnorm(out + res, block._post_ln_w, block._post_ln_eps)
        mlp = block.mlp
        if mlp._is_moe:
            mlp_out = moe_forward(h2.reshape(-1, self._hidden), mlp._gate_w, mlp._e_gu, mlp._e_d,
                                  self._top_k, self._n_experts, mlp._shared_gu, mlp._shared_d,
                                  decode=False)
            mlp_out = mlp_out.view(B, S, self._hidden)
        else:
            gate_up = h2 @ mlp._dense_gu
            gate, up = gate_up.chunk(2, dim=-1)
            mlp_out = (F.silu(gate) * up) @ mlp._dense_d
        return mlp_out + out + res

    def _build_prefill_slots(self, cache_manager, block_table, B, S):
        """构造 prefill 的 slot_mapping [B*S]：token (b, t) → block_table[b, t//block_size]*block_size + t%block_size"""
        bs = cache_manager.block_size
        bt = block_table[:B].long()                      # [B, max_seq_blocks]
        n_blocks = (S + bs - 1) // bs
        # 每 token 的 block 索引和 offset
        t = torch.arange(S, device=bt.device)
        block_idx = t // bs                              # [S]
        offset = t % bs
        slot = bt[:, :n_blocks][:, block_idx] * bs + offset  # [B, S]
        return slot.reshape(-1).to(torch.int32)

    @staticmethod
    def _decode_slots(block_table, new_pos, bs, block_size):
        """decode 单 token 的物理 slot [bs, 1]（int32）。
        new_pos: [bs] 逻辑位置。slot = block_table[seq, new_pos//block_size]*block_size + new_pos%block_size。"""
        bt = block_table[:bs].long()                     # [bs, max_seq_blocks]
        max_blk = bt.shape[1]
        block_idx = (new_pos // block_size).long().clamp(min=0, max=max_blk - 1)  # [bs]
        offset = (new_pos % block_size).long()           # [bs]
        # gather 每 seq 对应 block 的 id
        block_id = bt.gather(1, block_idx.unsqueeze(1)).squeeze(1)  # [bs]
        slot = block_id.clamp(min=0) * block_size + offset  # [bs]（防御 -1 非法 block_id）
        return slot.to(torch.int32).view(bs, 1)

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, hidden_dim, dtype=dtype, device=device),  # 占位
            "_attn_out": torch.empty(max_bs, hidden_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
        }
