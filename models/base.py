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
                   prefill_layer (prefill)

🔑 **关键约定**:
    - cache 维度由 `cache_dims()` 决定，直接喂给 KVCacheManager。
      GQA 模型返回 (num_heads, kv_num_heads, head_size)；
      MLA 模型返回 (1, 1, latent_dim) —— 把压缩 latent 当成 1 个 head 存储。
    - runner 不再硬编码任何模型字段名，全部经 adapter 访问。
"""
from abc import ABC, abstractmethod
from typing import Tuple, List
import torch


class ModelAdapter(ABC):
    model_type: str = ""

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @abstractmethod
    def cache_dims(self, cfg) -> Tuple[int, int, int]:
        """返回 (n_heads, kv_n_heads, head_size) 用于 KVCacheManager 分配。"""
        ...

    @abstractmethod
    def intermediate_size(self, cfg, world_size: int) -> int:
        """单卡 intermediate_size（已按 TP 切分）。"""
        ...

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
    @abstractmethod
    def embed(self, model) -> torch.Tensor:
        """返回 embedding 查表（callable: input_ids → hidden）。"""
        ...

    @abstractmethod
    def blocks(self, model) -> List:
        """返回 decoder layer 列表。"""
        ...

    @abstractmethod
    def final_norm(self, model):
        """返回最终 RMSNorm 层。"""
        ...

    @abstractmethod
    def lm_head(self, model):
        """返回 lm_head（callable: hidden → logits）。"""
        ...

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
    # prefill 单层钩子
    # ------------------------------------------------------------------
    @abstractmethod
    def prefill_layer(self, block, h: torch.Tensor, layer_idx: int,
                      B: int, S: int, graph, cache_manager, block_table) -> torch.Tensor:
        """prefill 单层前向（含 attention 写入 paged cache + FFN），返回新 hidden。"""
        ...

    # ------------------------------------------------------------------
    # buffer 分配（runner 在 _alloc_bufs 调用）
    # ------------------------------------------------------------------
    @abstractmethod
    def alloc_bufs(self, model, max_bs: int, hidden_dim: int, dtype, device) -> dict:
        """返回 runner 需要的中间 buffer 字典。"""
        ...

    def get_buf(self, bufs, name, bs):
        return bufs[name][:bs]
