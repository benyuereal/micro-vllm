"""
===================================================================
ModelAdapter - 多架构适配器抽象基类
===================================================================

📌 **设计目标**:
    把不同模型架构（Qwen / DeepSeek ...）的差异从推理 runner 中剥离。
    runner 只持有 `self.adapter`，通过统一钩子接口驱动任意架构，
    自身保持单一实现（CUDA Graph / prefill 框架 / 调度不变）。

🧩 **适配器职责**:
    1. 元信息：cache 维度、intermediate_size、层数
    2. 权重预处理：把原始 block 权重重排为 runner 复用的内部张量
    3. 模块访问：统一 embedding / layers / final_norm / lm_head 命名
    4. 单层前向钩子：compute_qkv / attention / compute_ffn (decode)
                   prefill (prefill)

🔑 **关键约定**:
    - cache 维度由 `cache_dims()` 决定，直接喂给 KVCacheManager。
      GQA 模型返回 (num_heads, kv_num_heads, head_size)；
      MLA 模型返回 (1, 1, latent_dim) —— 把压缩 latent 当成 1 个 head 存储。
    - runner 不再硬编码任何模型字段名，全部经 adapter 访问。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
import torch


@dataclass
class PrefillMeta:
    """变长 prefill 的批元数据（同一 batch 内各 seq 长度可不同，cu_seqlens 掩码处理）。

    约定：所有 seq 的本 chunk token 拼成 1D（按 seq 顺序），total_tokens = cu_seqlens_q[-1]。
    - cu_seqlens_q[c:c+1] = 第 c 条 seq 本 chunk 的 query 长度（= 本 chunk token 数）
    - cu_seqlens_k[c:c+1] = 第 c 条 seq 的 KV 长度（已 cache 前缀 + 本 chunk；完整 prefill 时 = q 长度）
    - position_ids[t] = token t 在原 prompt 的绝对位置（chunked 续写从 prefill_done 起）
    - slot_mapping[t] = token t 写入 paged cache 的 slot（block_id*block_size + offset）
    - block_table[c] = 第 c 条 seq 的 block id 列表（flash varlen 用其读 paged cache）
    """
    cu_seqlens_q: torch.Tensor   # [n_seqs+1] int32
    cu_seqlens_k: torch.Tensor   # [n_seqs+1] int32
    position_ids: torch.Tensor   # [total_tokens] long
    slot_mapping: torch.Tensor   # [total_tokens] int32
    block_table: torch.Tensor    # [n_seqs, max_seq_blocks] int32
    n_seqs: int
    max_seqlen_q: int
    max_seqlen_k: int


class ModelAdapter(ABC):
    model_type: str = ""

    # decode attention 路径开关：True=prerope+store+pure-flash（Qwen3 用，省 50us/层）；
    # False=flash internal-rotary+k=/v=（DeepSeek MLA 等）。基类默认 False，
    # 调用方直接读 self.adapter.use_prerope_decode，无需 getattr 兜底。
    use_prerope_decode = False

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def cache_dims(self, cfg) -> Tuple[int, int, int]:
        """返回 (n_heads, kv_n_heads, head_size) 用于 KVCacheManager 分配。
        GQA 通用默认：head_size 取 cfg.head_dim，否则 hidden/heads。MLA 等特殊架构 override。"""
        num_heads = cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_size = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
        return num_heads, kv_heads, head_size

    def intermediate_size(self, cfg, world_size: int) -> int:
        """单卡 intermediate_size（已按 TP 切分）。MoE 等特殊架构 override。"""
        return cfg.intermediate_size // world_size

    def num_layers(self, cfg) -> int:
        """decoder 层数。标准模型即 cfg.num_hidden_layers。"""
        return cfg.num_hidden_layers

    def rope_dim(self, cfg) -> int:
        """RoPE 实际作用的维度。GQA = head_size；MLA = qk_rope_head_dim。"""
        return self.cache_dims(cfg)[2]

    def rope_theta(self, cfg) -> float:
        """RoPE base frequency。默认 10000；Qwen3=1e6 等。"""
        return getattr(cfg, "rope_theta", None) or 10000.0

    def softmax_scale(self, cfg) -> float:
        """attention softmax 缩放。"""
        head_dim = self.cache_dims(cfg)[2]
        return head_dim ** -0.5

    def supports_chunked_prefill(self, cfg) -> bool:
        """该架构的 prefill 是否支持 chunked 续写（第 N chunk 的 attention 能读到
        cache 中前 N-1 chunk 的 KV）。GQA + flash_attn_with_kvcache 支持；MLA prefill
        用 flash_attn_func（自包含，不读 cache 前缀）暂不支持。默认 False（保守）。"""
        return False

    def context_length_limit(self, cfg) -> int:
        """该架构支持的上下文长度硬上限（cos/sin 表覆盖范围 / kernel 静态 shape 约束）。
        返回 None 表示无架构侧限制，由 engine 构造参数 max_context_length 决定。
        MLA 等 max_len 进静态 shape 的架构 override 为固定值（如 1024）。"""
        return None

    # ------------------------------------------------------------------
    # 权重预处理
    # ------------------------------------------------------------------
    @abstractmethod
    def prepare_weights(self, model, world_size: int, rank: int):
        """把各 block 权重重排为 runner 复用的内部张量（如 _qkv_w / _o / _gu / _d）。
        完成后应释放原始 nn.Linear 权重以省显存。幂等：重复调用应跳过。"""
        ...

    # ------------------------------------------------------------------
    # 模块访问（统一命名）
    # ------------------------------------------------------------------
    # HF 风格默认实现（model.model.{embed_tokens,layers,norm} + model.lm_head）。
    # 老 Qwen 用 model.transformer.* 命名，override 这几个方法。
    def embed(self, model) -> torch.Tensor:
        """返回 embedding 查表（callable: input_ids → hidden）。"""
        return model.model.embed_tokens

    def blocks(self, model) -> List:
        """返回 decoder layer 列表。"""
        return model.model.layers

    def final_norm(self, model):
        """返回最终 RMSNorm 层。"""
        return model.model.norm

    def final_norm_one_centered(self) -> bool:
        """final_norm 是否 1-centered（out = x*rrms*(1+w)）。Qwen3.5 是，Qwen3/DeepSeek 否。
        model_graph 的 decode 末层据此选 rmsnorm 的 one_centered 参数。"""
        return False

    def lm_head(self, model):
        """返回 lm_head（callable: hidden → logits）。"""
        return model.lm_head

    # ------------------------------------------------------------------
    # decode 单层钩子
    # ------------------------------------------------------------------
    @abstractmethod
    def compute_qkv(self, block, h: torch.Tensor, graph, bs: int) -> torch.Tensor:
        """对当前 hidden 做 input_layernorm（+ Q/K/V 投影，视架构），返回 attention 输入。"""
        ...

    @abstractmethod
    def compute_next_qkv(self, block_next, mlp_out_prev: torch.Tensor, res_prev: torch.Tensor,
                         graph, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """融合：下一层的 post-norm(residual)（+ QKV，视架构）。返回 (attn_input, new_residual)。"""
        ...

    @abstractmethod
    def attention(self, attn_input, block, layer_idx: int, bs: int,
                  graph, cache_manager, block_table) -> torch.Tensor:
        """对 attention 输入做 paged attention + O 投影，返回 attn_out [bs, hidden]。"""
        ...

    @abstractmethod
    def compute_ffn(self, block, attn_out: torch.Tensor, residual: torch.Tensor,
                    graph, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """post_attention_layernorm + FFN，返回 (mlp_out, new_residual)。"""
        ...

    # ------------------------------------------------------------------
    # prefill 单层钩子（变长：h 为 1D [total_tokens, hidden]，各 seq 长度由 meta.cu_seqlens 掩码）
    # ------------------------------------------------------------------
    @abstractmethod
    def prefill(self, block, h: torch.Tensor, layer_idx: int,
                graph, cache_manager, meta: "PrefillMeta") -> torch.Tensor:
        """变长 prefill 单层前向（含 attention 写入 paged cache + FFN），返回新 hidden [total_tokens, hidden]。"""
        ...

    # ------------------------------------------------------------------
    # buffer 分配（runner 在 _alloc_bufs 调用）
    # ------------------------------------------------------------------
    @abstractmethod
    def alloc_bufs(self, model, max_bs: int, hidden_dim: int, dtype, device) -> dict:
        """返回 runner 需要的中间 buffer 字典。"""
        ...

    # ------------------------------------------------------------------
    # 有状态层（GDN 线性注意力等）的 batch 元信息钩子
    # ------------------------------------------------------------------
    def gdn_stateful(self) -> bool:
        """该架构是否有 per-seq 递归状态（GDN）。True 时 engine 在每步 forward 前
        调 on_decode_batch / on_prefill_batch 把 batch 的 seq_id 传给 adapter，
        用于索引 per-seq 状态池（decode pad 行去重、prefill 首 chunk 清零状态）。"""
        return False

    def on_decode_batch(self, batch, graph):
        """decode 步 forward 前调用：batch 为 pad 后的序列列表（循环复制填充）。
        adapter 据此填 graph 上的状态池索引（真实行 vs pad 行）。默认 no-op。"""
        pass

    def on_prefill_batch(self, batch, graph):
        """prefill 步 forward 前调用：batch 为本 prefill batch 的序列列表。
        adapter 据此填状态池索引 + 首 chunk 清零状态。默认 no-op。"""
        pass

    def on_seq_finished(self, seq):
        """seq 完成（EOS/stop/max_tokens）时调用：释放其 per-seq 状态池 slot
        （GDN 等）。默认 no-op。"""
        pass
