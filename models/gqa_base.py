"""GQAAdapter - GQA 架构公共基类（Qwen3 / Qwen3.5 去重）。

Qwen3（GQA + QK-Norm，纯 bf16）与 Qwen3.5（GDN 混合 + full attention，W8A16/Marlin
int8）都继承 ModelAdapter，实现相同钩子。两者大量骨架相同，本基类抽出公共部分：

- _ln_eps：HF RMSNorm 两种 eps 属性名兼容（完全相同）。
- flash_attn import 兜底（完全相同）。
- prepare_weights 的 bf16 [N,K] 布局版（Qwen3 直接用；Qwen3.5 override 扩展 int8/GDN）。
- compute_qkv / compute_next_qkv / compute_ffn 的公共骨架（norm + 投影 + swiglu，
  线性层 / norm 变体留虚方法）。
- alloc_bufs 公共版（_h_buf/_qkv/_attn_out/_residual，qkv 维留虚方法）。
- attention 的 prerope+store+flash 公共路径（QK-Norm+RoPE / k 段偏移 / attn gate 留虚方法）。

关键约束：Qwen3.5 有 W8A16/Marlin int8 路径（_lin/_lin_prefill/_store_w/_to_marlin/
_unpack_linear 等），Qwen3 是纯 bf16。基类把「线性层调用」和「norm 变体」抽象成虚方法，
int8 路径全部留在 Qwen3.5 的 override 里，不破坏。

差异点（留虚方法 / 子类 override）：
- norm 变体：Qwen3 标准 rmsnorm，Qwen3.5 用 one_centered=True（1-centered）。
- 线性分派：Qwen3 用 gemv_or_matmul（bf16），Qwen3.5 用 _lin（bf16/int8/Marlin）。
- QK-Norm+RoPE kernel：Qwen3 用 qk_norm_rope_inplace（全 head_dim），
  Qwen3.5 用 qk_norm_rope_partial_inplace（partial rot，1-centered）。
- qkv 布局：Qwen3 [q|k|v]（k 紧跟 q），Qwen3.5 [q|gate|k|v]（k 偏移 2*q_dim）。
- attn gate：Qwen3 无，Qwen3.5 有（attn *= sigmoid(gate)）。
- GDN：Qwen3.5 有 GDN 层（compute_qkv/attention/prefill/alloc_bufs override），Qwen3 无。
"""
import torch

from models.base import ModelAdapter
from kernel.rmsnorm import rmsnorm, rmsnorm_residual
from kernel.dense_mlp import dense_swiglu
from kernel.gemv import gemv_or_matmul
from core.cache_manager import store_kvcache

try:
    from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
except ImportError:
    flash_attn_with_kvcache = None
    flash_attn_varlen_func = None


class GQAAdapter(ModelAdapter):
    """GQA 架构公共基类。Qwen3 / Qwen3.5 继承并 override 差异点。"""

    # decode attention 走 prerope+store+pure-flash（两者都用）。
    use_prerope_decode = True

    # -------------------- 元信息 --------------------
    @staticmethod
    def _ln_eps(ln, cfg):
        """兼容 HF RMSNorm 的两种 eps 属性名（eps / variance_epsilon），回退 cfg.rms_norm_eps。"""
        return getattr(ln, "eps", None) or getattr(ln, "variance_epsilon", cfg.rms_norm_eps)

    # -------------------- 虚方法：norm 变体 --------------------
    def _norm_inplace(self, h, w, out, eps):
        """input_layernorm（decode，写预分配 buffer）。Qwen3 标准，Qwen3.5 override 为 1-centered。"""
        rmsnorm(h, w, eps, out=out)

    def _norm_residual(self, x, res, w, out_normed, out_residual, eps):
        """post-norm(residual) 贴边融合（decode）。Qwen3 标准，Qwen3.5 override 为 1-centered。"""
        rmsnorm_residual(x, res, w, eps, out_normed=out_normed, out_residual=out_residual)

    # -------------------- 虚方法：线性分派 --------------------
    def _lin(self, x, w, out, env="MICRO_GEMV"):
        """decode 线性（M=1 GEMV / M>1 matmul）。Qwen3=gemv_or_matmul（bf16）；
        Qwen3.5 override 为 bf16/int8/Marlin 统一分派。out 预分配。"""
        return gemv_or_matmul(x, w, out, env)

    def _lin_prefill(self, x, w):
        """prefill 线性（M=T>1）。Qwen3=x @ w.t()（w [N,K]）；Qwen3.5 override 为 int8/Marlin。"""
        return torch.matmul(x, w.t())

    # -------------------- 虚方法：GDN 分支（Qwen3.5 有 GDN 层，Qwen3 无） --------------------
    def _block_is_gdn(self, block) -> bool:
        """该层是否为 GDN 线性注意力层。Qwen3 恒 False；Qwen3.5 按 block._is_gdn。"""
        return False

    def _attention_gdn_decode(self, h_normed, block, bs, graph):
        """GDN 层 decode（Qwen3.5 override）。Qwen3 无 GDN，不实现。"""
        raise NotImplementedError

    # -------------------- 虚方法：qkv 布局 --------------------
    def _qkv_dim(self, model) -> int:
        """qkv buffer 的列数（alloc_bufs 用）。Qwen3 读 _qkv_w.shape[0]；
        Qwen3.5 权重可能是 Marlin dict（无 .shape），按 config 计算。"""
        return self.blocks(model)[0].self_attn._qkv_w.shape[0]

    def _k_offset(self, graph) -> int:
        """qkv buffer 里 k 段的起始列。Qwen3 [q|k|v] → q_dim；Qwen3.5 [q|gate|k|v] → 2*q_dim。"""
        return graph.num_heads * graph.head_size

    def _qk_norm_rope(self, qkv, bs, seg_offset, num_heads, head_size,
                      norm_w, cos_pool, sin_pool, cache_lens, eps):
        """对 qkv 某段原地做 QK-Norm + RoPE（decode prerope 路径）。
        Qwen3=qk_norm_rope_inplace（全 head_dim），Qwen3.5=qk_norm_rope_partial_inplace（partial）。"""
        raise NotImplementedError

    def _apply_attn_gate(self, attn, qkv, graph, bs):
        """attn_output_gate（in-place attn *= sigmoid(gate)）。Qwen3 无（no-op），Qwen3.5 有。"""
        pass

    # -------------------- 权重预处理（bf16 [N,K] 布局版） --------------------
    def _prepare_ln(self, block, cfg):
        """input/post attention layernorm 权重 + eps（两者相同）。"""
        block._in_ln_w = block.input_layernorm.weight.data.clone()
        block._in_ln_eps = self._ln_eps(block.input_layernorm, cfg)
        block._post_ln_w = block.post_attention_layernorm.weight.data.clone()
        block._post_ln_eps = self._ln_eps(block.post_attention_layernorm, cfg)

    @staticmethod
    def _release_attn(attn):
        """释放 attention 原始权重（省显存）。"""
        attn.q_proj = attn.k_proj = attn.v_proj = attn.o_proj = None
        attn.q_norm = attn.k_norm = None

    @staticmethod
    def _release_mlp(mlp):
        """释放 MLP 原始权重（省显存）。"""
        mlp.gate_proj = mlp.up_proj = mlp.down_proj = None

    def prepare_weights(self, model, world_size, rank):
        """bf16 [N,K] 布局版（Qwen3 直接用）。Qwen3.5 override 扩展 int8/GDN/lm_head。

        权重统一存 [N,K]=[out,in]（HF 原始布局，不 .t()）：GEMV 友好（每输出行连续 K），
        手写 gemv_v2 kernel 直接读；torch.matmul 路径用 W.t()（opT，裸 cuBLAS 无损）。
        """
        first = self.blocks(model)[0]
        if getattr(first.self_attn, "_prepared", False):
            return
        cfg = model.config

        for block in self.blocks(model):
            attn = block.self_attn
            mlp = block.mlp
            # Q/K/V：HF [out,in]，cat 沿输出维 dim=0 → [q_dim+2*kv_dim, hidden]
            w_q = attn.q_proj.weight.data.chunk(world_size, dim=0)[rank]  # [q_dim, hidden]
            w_k = attn.k_proj.weight.data.chunk(world_size, dim=0)[rank]  # [kv_dim, hidden]
            w_v = attn.v_proj.weight.data.chunk(world_size, dim=0)[rank]  # [kv_dim, hidden]
            attn._qkv_w = torch.cat([w_q, w_k, w_v], dim=0).contiguous()  # [q_dim+2*kv_dim, hidden]
            attn._qkv_b = None  # attention_bias=false
            # O 投影：[hidden, q_dim]（HF o_proj.weight = [out=hidden, in=q_dim]）
            attn._o_w = attn.o_proj.weight.data.chunk(world_size, dim=1)[rank].contiguous()
            # QK-Norm 权重（RMSNorm on head_dim，shape [head_dim]）
            attn._q_norm_w = attn.q_norm.weight.data.clone()
            attn._k_norm_w = attn.k_norm.weight.data.clone()
            attn._q_norm_eps = self._ln_eps(attn.q_norm, cfg)
            attn._k_norm_eps = self._ln_eps(attn.k_norm, cfg)

            # MLP: dense_swiglu 约定 gu_w 输出维 [up|gate] 顺序（前半 up、后半 gate）。
            # [N,K] 布局：cat 沿输出维 dim=0 → [2*inter, hidden]，前半 up、后半 gate。
            w_up = mlp.up_proj.weight.data.chunk(world_size, dim=0)[rank]    # [inter, hidden]
            w_gate = mlp.gate_proj.weight.data.chunk(world_size, dim=0)[rank]  # [inter, hidden]
            mlp._gu = torch.cat([w_up, w_gate], dim=0).contiguous()  # [2*inter, hidden]
            mlp._d = mlp.down_proj.weight.data.chunk(world_size, dim=1)[rank].contiguous()  # [hidden, inter]

            self._prepare_ln(block, cfg)
            self._release_attn(attn)
            self._release_mlp(mlp)
            attn._prepared = True
        torch.cuda.empty_cache()

    # -------------------- decode 单层钩子 --------------------
    def _project_qkv(self, attn, graph, bs):
        """normed h（graph._h_buf[:bs]）→ QKV 投影，写 graph._qkv[:bs]。"""
        qkv_buf = graph._qkv[:bs]
        self._lin(graph._h_buf[:bs], attn._qkv_w, qkv_buf, "MICRO_GEMV_QKV")
        return qkv_buf

    def compute_qkv(self, block, h, graph, bs):
        self._norm_inplace(h, block._in_ln_w, graph._h_buf[:bs], block._in_ln_eps)
        if self._block_is_gdn(block):
            # GDN 层：attn_input = 归一化后的 h（投影延迟到 attention 内做）。
            return graph._h_buf[:bs]
        return self._project_qkv(block.self_attn, graph, bs)

    def compute_next_qkv(self, block_next, mlp_out_prev, res_prev, graph, bs):
        # 返回 (attn_input, residual)：decode 循环 `qkv, h = compute_next_qkv(...)` 解包两值。
        # GDN 层 attn_input = 归一化后的 h（投影延迟到 attention 内做）；full 层 = 投影后 qkv。
        self._norm_residual(
            mlp_out_prev, res_prev, block_next._in_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block_next._in_ln_eps
        )
        if self._block_is_gdn(block_next):
            return graph._h_buf[:bs], graph._residual[:bs]
        return self._project_qkv(block_next.self_attn, graph, bs), graph._residual[:bs]

    def compute_ffn(self, block, attn_out, residual, graph, bs):
        self._norm_residual(
            attn_out, residual, block._post_ln_w,
            graph._h_buf[:bs], graph._residual[:bs], block._post_ln_eps
        )
        # dense_swiglu：权重 [N,K] 布局（w_is_nk=True），M=1 decode 走 gemv_v2；
        # int8/Marlin 权重由 dense_swiglu 内部 _lin 分派处理。
        mlp_out = dense_swiglu(graph._h_buf[:bs], block.mlp._gu, block.mlp._d, bs, w_is_nk=True)
        return mlp_out, graph._residual[:bs]

    # -------------------- attention（prerope+store+flash 公共路径） --------------------
    def attention(self, attn_input, block, layer_idx, bs, graph, cache_manager, block_table):
        if self._block_is_gdn(block):
            return self._attention_gdn_decode(attn_input, block, bs, graph)
        return self._attention_full_decode(attn_input, block, layer_idx, bs,
                                           graph, cache_manager, block_table)

    def _attention_full_decode(self, qkv, block, layer_idx, bs, graph,
                               cache_manager, block_table):
        """full attention decode：QK-Norm+RoPE（prerope）→ store k/v → pure flash →
        attn gate → O 投影。QK-Norm+RoPE kernel / k 段偏移 / attn gate 留虚方法。"""
        sa = block.self_attn
        nh, kvh, hd = graph.num_heads, graph.kv_num_heads, graph.head_size
        q_dim = nh * hd
        kv_dim = kvh * hd
        k_off = self._k_offset(graph)
        cache_lens = cache_manager._cache_seqlens_buffer[:bs]
        cos_pool = graph.attention._cos_pool
        sin_pool = graph.attention._sin_pool

        # QK-Norm + RoPE 融合（prerope 路径）：对 q 段、k 段原地 norm+rotate。
        self._qk_norm_rope(qkv, bs, 0, nh, hd, sa._q_norm_w,
                           cos_pool, sin_pool, cache_lens, sa._q_norm_eps)
        self._qk_norm_rope(qkv, bs, k_off, kvh, hd, sa._k_norm_w,
                           cos_pool, sin_pool, cache_lens, sa._k_norm_eps)

        q = qkv[:, :q_dim].view(bs, nh, hd)
        k = qkv[:, k_off:k_off + kv_dim].view(bs, kvh, hd)
        v = qkv[:, k_off + kv_dim:].view(bs, kvh, hd)

        k_cache, v_cache = cache_manager.get(layer_idx)
        store_kvcache(k, v, k_cache, v_cache, graph._slot_mapping[:bs])
        # flash 读 cache_seqlens+1（含刚 store 的当前 token）
        attn = flash_attn_with_kvcache(
            q=q.unsqueeze(1), k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=graph._flash_seqlens[:bs], block_table=block_table,
            causal=True, window_size=(-1, -1), alibi_slopes=None,
            num_splits=0 if bs == 1 else (0 if bs >= 32 else max(1, 32 // max(1, bs * 4)))
        ).squeeze(1)

        self._apply_attn_gate(attn, qkv, graph, bs)

        out_buf = graph._attn_out[:bs]
        return self._lin(attn.reshape(bs, -1), sa._o_w, out_buf, "MICRO_GEMV_O")

    # -------------------- buffer 分配 --------------------
    def alloc_bufs(self, model, max_bs, hidden_dim, dtype, device):
        """公共 4 buffer（_h_buf/_qkv/_attn_out/_residual）。Qwen3.5 override 追加 GDN 状态池。"""
        return {
            "_h_buf": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
            "_qkv": torch.empty(max_bs, self._qkv_dim(model), dtype=dtype, device=device),
            "_attn_out": torch.empty(max_bs, hidden_dim, dtype=dtype, device=device),
            "_residual": torch.empty((max_bs, hidden_dim), dtype=dtype, device=device),
        }
