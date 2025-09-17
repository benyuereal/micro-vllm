import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLayerAdapter:
    """
    📌 **模型层适配器** - vLLM核心组件

    🔍 **设计哲学**:
        1. **统一接口**: 所有模型架构使用相同的process_layer接口
        2. **自动适配**: 根据model_type自动选择处理逻辑
        3. **零拷贝**: 直接操作张量，无中间拷贝
        4. **生产就绪**: 支持AMP、异常处理、设备匹配

    🧪 **典型用法**:
        adapter = ModelLayerAdapter(config, device, num_heads=16, head_size=128, kv_num_heads=16)
        hidden_states, (k, v) = adapter.process_layer(
            layer=layer,
            hidden_states=hidden_states,  # [B, S, D]
            cache_manager=cache_manager,  # KVCacheManager实例
            seq_ids=[0, 1, 2],          # 序列ID列表
            context_lens=[10, 20, 30],   # 当前长度
            token_positions=positions,   # token位置 (可选)
            layer_idx=0,                 # 层索引
            current_positions=positions  # 当前位置 (可选)
        )
    """

    # 模型架构配置 (可扩展)
    MODEL_CONFIGS = {
        "qwen": {  # Qwen 7B
            "norm": "ln_1", "attn": "c_attn", "proj": "c_proj", "mlp_norm": "ln_2",
            "qkv_split": True, "qkv_proj": False,
            "mlp": "mlp", "residual": True,
        },
        "qwen2": {  # Qwen 1.5/2.5
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
        },
        "qwen3": {  # Qwen3 (与Qwen2相同，但支持MoE)
            "norm": "input_layernorm", "attn": None, "proj": "o_proj", "mlp_norm": "post_attention_layernorm",
            "qkv_split": False, "qkv_proj": True,
            "mlp": "mlp", "residual": True,
            "moe": True,  # ✅ 支持MoE
        },
    }

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int):
        """
        📌 **初始化**

        🔍 **参数**:
            - model_config: 模型配置
            - device: 设备 ("cuda", "mps", "cpu")
            - num_heads: 注意力头数
            - head_size: 每个头维度
            - kv_num_heads: KV头数 (GQA支持)
        """
        self.config = model_config
        self.device = device
        self.model_type = model_config.model_type
        self.num_heads, self.head_size, self.kv_num_heads = num_heads, head_size, kv_num_heads

        # 性能统计
        self.layer_times = []
        self.total_calls = 0
        self.enable_profiling = True  # 是否启用性能分析

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=kv_num_heads,
            device=device
        )

        # 验证模型类型
        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.cfg = self.MODEL_CONFIGS[self.model_type]

        logger.info(f"ModelLayerAdapter initialized for {self.model_type} on {device}")

    def _log_timing(self, stage_name: str, start_time: float, layer_idx: int, batch_size: int, seq_len: int):
        """记录微秒级别的时间消耗"""
        if not self.enable_profiling:
            return

        elapsed_us = (time.time() - start_time) * 1e6  # 转换为微秒
        logger.debug(f"Layer {layer_idx} | {stage_name}: {elapsed_us:.2f}μs "
                    f"(batch={batch_size}, seq={seq_len})")
        return elapsed_us

    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.layer_times:
            return "No performance data available"

        import numpy as np
        times = np.array(self.layer_times)

        stats = {
            'total_calls': self.total_calls,
            'avg_total_time_us': np.mean(times[:, 0]),
            'avg_norm_time_us': np.mean(times[:, 1]),
            'avg_qkv_time_us': np.mean(times[:, 2]),
            'avg_reshape_time_us': np.mean(times[:, 3]),
            'avg_attention_time_us': np.mean(times[:, 4]),
            'avg_proj_time_us': np.mean(times[:, 5]),
            'avg_mlp_time_us': np.mean(times[:, 6]),
            'p95_total_time_us': np.percentile(times[:, 0], 95)
        }

        return stats

    def process_layer(self,
                      layer,
                      hidden_states: torch.Tensor,  # [B, S, D]
                      cache_manager,
                      seq_ids: List[int],
                      context_lens: List[int],
                      token_positions: Optional[torch.Tensor] = None,
                      layer_idx: int = 0,
                      current_positions: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        📌 **处理单层计算** (统一接口，自动适配模型架构)

        🔍 **参数**:
            - layer: 模型层 (transformer layer)
            - hidden_states: 隐藏状态 [B, S, D]
            - cache_manager: KVCacheManager实例
            - seq_ids: 序列ID列表 [B]
            - context_lens: 当前长度列表 [B]
            - token_positions: token位置 (可选)
            - layer_idx: 层索引
            - current_positions: 当前位置 (可选)

        ✅ **返回**:
            - hidden_states: 更新后的隐藏状态 [B, S, D]
            - (current_k, current_v): 当前层的KV [B, H, D]

        🧠 **内部逻辑**:
            1. 自动适配模型架构 (Qwen/Qwen2等)
            2. 应用LayerNorm
            3. 计算QKV (自动处理不同投影方式)
            4. 重塑形状 [B, S, D] → [B, H, D]
            5. 调用PagedAttention
            6. 残差连接 + MLP
        """
        total_start = time.time()
        batch_size, seq_len, _ = hidden_states.shape
        stage_times = [0.0] * 7  # 存储各阶段耗时

        # 记录批次和序列信息
        if self.total_calls % 100 == 0:  # 每100次记录一次详细信息
            logger.info(f"Processing layer {layer_idx} | batch={batch_size}, seq={seq_len}")

        # 1. 自动适配模型架构
        norm_fn = getattr(layer, self.cfg["norm"])
        mlp_norm_fn = getattr(layer, self.cfg["mlp_norm"])
        mlp_fn = getattr(layer, self.cfg["mlp"])

        # 2. LayerNorm + 残差
        norm_start = time.time()
        residual = hidden_states
        hidden_states = norm_fn(hidden_states)
        stage_times[1] = self._log_timing("LayerNorm", norm_start, layer_idx, batch_size, seq_len)

        # 3. QKV计算 (自动处理不同投影方式)
        qkv_start = time.time()
        if self.cfg["qkv_split"]:
            # Qwen 7B: 合并的c_attn投影
            qkv = layer.attn.c_attn(hidden_states)
            hidden_size = qkv.shape[-1] // 3
            q, k, v = qkv.split(hidden_size, dim=-1)
        else:
            # Qwen 1.5: 分开的q_proj/k_proj/v_proj
            q = layer.self_attn.q_proj(hidden_states)
            k = layer.self_attn.k_proj(hidden_states)
            v = layer.self_attn.v_proj(hidden_states)
        stage_times[2] = self._log_timing("QKV_Projection", qkv_start, layer_idx, batch_size, seq_len)

        # 4. 重塑形状 [B, S, D] → [B, H, D]
        reshape_start = time.time()
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        stage_times[3] = self._log_timing("Reshape", reshape_start, layer_idx, batch_size, seq_len)

        # 5. 注意力计算 (零拷贝)
        attention_start = time.time()
        attn_output = self.attention(
            query=q.squeeze(2),  # [B, H, D]
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=k.squeeze(2),  # [B, H, D]
            value=v.squeeze(2)  # [B, H, D]
        )
        stage_times[4] = self._log_timing("Attention", attention_start, layer_idx, batch_size, seq_len)

        # 6. 输出投影 + 残差
        proj_start = time.time()
        proj_fn = getattr(layer.self_attn if self.cfg["qkv_proj"] else layer.attn, self.cfg["proj"])
        attn_output = proj_fn(attn_output.reshape(batch_size, -1)).unsqueeze(1)  # [B, 1, D]
        hidden_states = residual + attn_output
        stage_times[5] = self._log_timing("Projection+Residual", proj_start, layer_idx, batch_size, seq_len)

        # 7. MLP + 残差 (支持MoE)
        mlp_start = time.time()
        residual = hidden_states
        hidden_states = mlp_norm_fn(hidden_states)
        if self.cfg.get("moe", False):
            # ✅ Qwen3 MoE: 使用 mlp 模块 (包含 experts 和 gate)
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                hidden_states = layer.mlp(hidden_states)  # 直接调用mlp模块
        else:
            # Qwen2: 普通MLP
            hidden_states = mlp_fn(hidden_states)
        hidden_states = residual + hidden_states
        stage_times[6] = self._log_timing("MLP+Residual", mlp_start, layer_idx, batch_size, seq_len)

        # 记录总耗时
        total_time = self._log_timing("TOTAL_LAYER", total_start, layer_idx, batch_size, seq_len)
        stage_times[0] = total_time

        # 更新性能统计
        if self.enable_profiling:
            self.layer_times.append(stage_times)
            self.total_calls += 1

            # 每100层输出一次性能摘要
            if self.total_calls % 100 == 0:
                stats = self.get_performance_stats()
                logger.info(f"Performance Summary after {self.total_calls} layers:")
                logger.info(f"  Avg Total: {stats['avg_total_time_us']:.2f}μs")
                logger.info(f"  P95 Total: {stats['p95_total_time_us']:.2f}μs")
                logger.info(f"  Norm: {stats['avg_norm_time_us']:.2f}μs")
                logger.info(f"  QKV: {stats['avg_qkv_time_us']:.2f}μs")
                logger.info(f"  Reshape: {stats['avg_reshape_time_us']:.2f}μs")
                logger.info(f"  Attention: {stats['avg_attention_time_us']:.2f}μs")
                logger.info(f"  Projection: {stats['avg_proj_time_us']:.2f}μs")
                logger.info(f"  MLP: {stats['avg_mlp_time_us']:.2f}μs")

        return hidden_states, (k.squeeze(2), v.squeeze(2))  # [B, H, D]