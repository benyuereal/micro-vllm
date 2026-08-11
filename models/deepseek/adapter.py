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
# 融合 MLA decode kernel（latent→rmsnorm+RoPE+paged flash，weight-absorption）。
# 把 attention 的 gather+kvb+rope+cat+flash 压进单个 kernel。
from kernel.mla import _get_kernel as _get_mla_kernel
# 融合 MoE decode kernel（routed experts: gate_up+silu+down，M=16 grid-parallel）。
from kernel.moe import moe_decode
# pre-MLA 全融合 persistent kernel（pre_qkv∥pre_kva→absorb 单 kernel，替代 3 个独立 kernel）。
from kernel.pre_mla import get_premla_persistent_kernel

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


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
        # 固定上下文长度（与 model_graph._deepseek_fixed_maxlen 同一常量，两处均=1024）。
        # cos/sin 表只需覆盖此长度：decode new_pos ≤ 1023、prefill S ≤ 1024。
        self._max_pos = 1024

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
            # pre-kernel 需要非 None 的 bias 张量；DeepSeek q_proj/kva 无 bias → 零向量
            attn._q_b = attn.q_proj.bias.data.clone() if attn.q_proj.bias is not None else \
                torch.zeros(self._num_heads * self._q_head, dtype=attn._q_w.dtype, device=attn._q_w.device)
            # kv_a_proj_with_mqa: [kv_lora+qk_rope, hidden]
            attn._kva_w = attn.kv_a_proj_with_mqa.weight.data.clone()
            attn._kva_b = attn.kv_a_proj_with_mqa.bias.data.clone() if attn.kv_a_proj_with_mqa.bias is not None else \
                torch.zeros(self._latent_dim, dtype=attn._kva_w.dtype, device=attn._kva_w.device)
            # kv_a_layernorm weight: [kv_lora_rank]
            attn._kva_ln_w = attn.kv_a_layernorm.weight.data.clone()
            attn._kva_ln_eps = attn.kv_a_layernorm.variance_epsilon
            # kv_b_proj: [num_heads*(qk_nope+v_head), kv_lora_rank]
            attn._kvb_w = attn.kv_b_proj.weight.data.clone()
            # 预拆 per-head 的 kvb 权重供 MLA weight-absorption 用：
            #   _kvb_w_kn[h] = kvb_w[h*256 : h*256+128]   (吸收进 Q → A)
            #   _kvb_w_v[h]  = kvb_w[h*256+128 : h*256+256] (post-multiply → out)
            _kvb_full = attn._kvb_w.view(self._num_heads,
                                         self._qk_nope + self._v_head,
                                         self._kv_lora_rank)
            attn._kvb_w_kn = _kvb_full[:, :self._qk_nope, :].contiguous()
            attn._kvb_w_v = _kvb_full[:, self._qk_nope:, :].contiguous()
            # absorb kernel 需要转置权重 [H, kv_lora, qk_nope]（[H,k,d] 而非 [H,d,k]）
            attn._kvb_w_kn_t = attn._kvb_w_kn.transpose(1, 2).contiguous()
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

    def _build_rope_tables(self, dtype, device):
        """预计算 YaRN cos/sin 全宽表 [_max_pos, qk_rope]（cat(freqs,freqs)）。

        在 alloc_bufs 时一次性算好（runner __init__ 阶段，早于 capture），存进 bufs
        作为 runner 生命周期的常驻张量。消除旧 _rope_pool 的 lazy 计算——后者依赖
        "首次调用恰好落在 capture warmup 之外"的时序巧合，显式预计算更鲁棒。
        max_pos 只需覆盖固定上下文（1024），不取框架 RoPE 表全容量 8192（省 7× 行 + 显存）。
        """
        dim = self._qk_rope
        cfg = self._cfg_cache
        base = getattr(cfg, "rope_theta", 10000)
        scaling = getattr(cfg, "rope_scaling", None) or {}
        scaling_factor = scaling.get("factor", 1.0)
        original_max_pos = scaling.get("original_max_position_embeddings", 4096)
        beta_fast = scaling.get("beta_fast", 32)
        beta_slow = scaling.get("beta_slow", 1)
        max_pos = self._max_pos
        inv_freq = self._yarn_inv_freq(dim, base, scaling_factor, original_max_pos,
                                       beta_fast, beta_slow, device)
        t = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)            # [max_pos, dim]
        return emb.cos().to(dtype).contiguous(), emb.sin().to(dtype).contiguous()

    def _rope_tables(self, graph):
        """返回 alloc_bufs 预算好的 cos/sin 全宽表（纯 getter，无计算）。"""
        return graph._cos_full, graph._sin_full

    def _rope_pool(self, graph, device=None):
        """旧名兼容（tests/ 原型仍用 _rope_pool）；新代码用 _rope_tables。"""
        return self._rope_tables(graph)

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
        # input_layernorm → normed x 写进 _x16[:,0,:]（strided view，pre-MLA kernel 读 [bs,16,hidden] row 0）
        rmsnorm_(h, block._in_ln_w, graph._x16[:bs, 0, :], block._in_ln_eps)
        return graph._x16[:bs, 0, :]  # normed（residual 由调用方传）

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        # rmsnorm_residual：normed 写 _x16[:,0,:]，residual 写 _residual
        rmsnorm_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._x16[:bs, 0, :], graph._residual[:bs], block_next._in_ln_eps
        )
        return graph._x16[:bs, 0, :], graph._residual[:bs]

    def attention(self, x_normed, block, layer_idx, bs, graph, cache_manager, block_table):
        """decode MLA attention（pre-MLA 全融合 + MLA decode kernel）。

        契约：forward 时 cache_seqlens[i] = 当前序列“下一步”的预期长度（框架在 prefill 后把
        current_position 置为 S+1，故首步 decode seqlens=S+1）。新 token 的逻辑位置 =
        cache_seqlens - 1（0-indexed），写入该 slot，RoPE 用该位置，attention 覆盖
        [0, cache_seqlens-1]（共 cache_seqlens 个 token，全部有效，无空洞）。forward 后 commit +1。

        pre-MLA 全融合（kernel/pre_mla.py）替代原来的 PyTorch q_proj/kva_proj/store/rope/absorb：
          pre_qkv  : x16 @ q_w^T → q[bs,H,16,q_head]，q_pe 列在 epilogue 做 rope。
          pre_kva  : x16 @ kva_w^T → latent 直写 paged cache（store epilogue）。
          absorb   : q_nope @ kvb_w_kn_t → A[bs,H,kv_lora]。
        然后 MLA kernel（不变）读 A、qpe 做 rmsnorm+RoPE+paged flash。
        """
        attn = block.self_attn
        k_cache, v_cache = cache_manager.get(layer_idx)    # [n_blocks, block_size, 1, 576]
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]  # [bs] int32
        new_pos = (cache_lens - 1).clamp(min=0)            # 新 token 逻辑位置（0-indexed，int32 全程）
        # 防御：极少数竞态下首步 decode seqlens 可能为 0 → new_pos=-1 → gather 负索引崩溃。
        # 钳到 0 保证不崩（输出可能不准，但避免 device-side assert 拖垮整个 server）。
        # cache_lens 已 int32，clamp 在 int32 上安全；旧版 .long().to(int32) 是冗余 cast 链（每层 2 节点×27）。

        max_len = graph._cur_bucket_maxlen
        block_size = cache_manager.block_size
        cos, sin = self._rope_tables(graph)  # [max_pos, qk_rope] 全宽 cat(freqs,freqs)
        # cos/sin 的位置查找（cos[new_pos]）已移进 pre_qkv kernel 内部（省外部 gather+cast 节点）。

        # ---------- pre-MLA 全融合 persistent kernel ----------
        # pre_qkv ∥ pre_kva → absorb 折进单个 T.Kernel(NUM_SMS) persistent（+2.0%）。
        # kernel 内部按 new_pos 从 cos/sin 全池 gather（省外部 cos[new_pos].to(dtype)），
        # 并直写紧凑 QpeOut[bs, h, qk_rope]（省外部 q_pe slice+contiguous 拷贝）。
        x16 = graph._x16[:bs]                               # [bs, 16, hidden]，row 0 = normed x
        bt = block_table[:bs]                               # [bs, max_seq_blocks] view（已 contiguous）
        k_pers, q_out_p, q_pe = get_premla_persistent_kernel(
            bs, self._hidden, self._num_heads, self._q_head, self._qk_rope,
            self._qk_nope, self._kv_lora_rank, self._latent_dim, block_size,
            bt.shape[1], k_cache.shape[0], cos.shape[0], graph.dtype)
        # X16 = graph._x16[:bs]：row0 已由 compute_qkv 的 rmsnorm_ 预填 normed x，
        # rows1-15 恒零（alloc zeros）。kernel 直接读，无需 H_in/phase0 copy。
        A_in = k_pers(attn._q_w, attn._q_b, cos, sin, attn._kva_w, attn._kva_b,
                      attn._kvb_w_kn_t, graph._absorb_idx[:bs * self._num_heads],
                      x16, q_out_p, q_pe, bt, new_pos, k_cache, v_cache)
        A_in = A_in.reshape(bs, self._num_heads, self._kv_lora_rank)

        # ---------- 融合 MLA decode（不变）----------
        # cos/sin/k_cache/block_table/cache_lens 均已 contiguous，省 arange+indexing+.contiguous()
        # 的空 kernel（graph 下仍是节点）。k_pos=arange(max_len) 后 cos[k_pos] ≡ cos[:max_len]。
        cos_k = cos[:max_len]                              # [max_len, qk_rope] view
        sin_k = sin[:max_len]
        Latent_flat = k_cache.reshape(-1, 1, self._latent_dim)  # 已 contiguous，view 无拷贝
        n_slots = k_cache.shape[0] * block_size
        kernel = _get_mla_kernel(
            bs, self._num_heads, max_len, self._kv_lora_rank, self._qk_rope,
            self._qk_nope, self._v_head, block_size, graph._ds_softmax_scale,
            graph.dtype, n_slots, block_N=64, num_split=4)
        attn_out = kernel(
            A_in, q_pe, Latent_flat,
            block_table[:bs],
            cache_lens,
            attn._kva_ln_w, attn._kvb_w_v, cos_k, sin_k)
        attn_out = attn_out.reshape(bs, self._num_heads * self._v_head)
        return F.linear(attn_out, attn._o_w, attn._o_b)

    def compute_ffn(self, block, attn_out, residual, graph, bs, fast_mode):
        mlp = block.mlp
        rmsnorm_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        x = graph._h_buf[:bs]
        if mlp._is_moe:
            mlp_out = moe_decode(
                x, mlp._gate_w, mlp._e_gu, mlp._e_d,
                self._top_k, self._n_experts,
                mlp._shared_gu, mlp._shared_d,
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
        cos_full, sin_full = self._rope_tables(graph)
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
                                  self._top_k, self._n_experts, mlp._shared_gu, mlp._shared_d)
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

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        # YaRN cos/sin 全宽表：alloc_bufs 时一次性预算（早于 capture），runner 生命周期常驻。
        cos_full, sin_full = self._build_rope_tables(dtype, device)
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, hidden_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            # M=16 零填充的 normed x：pre-MLA kernel 读 [bs,16,hidden]，row 0 真实。
            # rmsnorm 直接写 _x16[:,0,:]（strided view），省一次 pad copy。rows 1-15 恒零
            # （GEMM 输出 row 0 独立于其余行，垃圾/零都不影响 row 0；零更安全）。
            "_x16": torch.zeros((max_bs, 16, hidden_dim), dtype=dtype, device=device),
            # absorb 的 head 索引缓冲（[bs*H] % H）
            "_absorb_idx": (torch.arange(max_bs * self._num_heads, dtype=torch.int32, device=device) % self._num_heads),
            # RoPE cos/sin 全宽表 [_max_pos, qk_rope]（预算好的常驻张量）
            "_cos_full": cos_full,
            "_sin_full": sin_full,
        }
