# File: optimized_qwen_layer.py
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
import logging
import time
from .gptq import GPTQCUDAFusion

logger = logging.getLogger(__name__)


class OptimizedQwenModelLayerAdapter:
    """
    优化的Qwen7B INT4量化模型适配器
    基于layer.py的高效实现，使用CUDA融合内核
    """

    def __init__(self, model_config, device: str, num_heads: int, head_size: int, kv_num_heads: int = None):
        self.config = model_config
        self.device = device
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_num_heads = kv_num_heads if kv_num_heads is not None else num_heads

        # 初始化注意力模块
        self.attention = PagedAttention(
            num_heads=num_heads,
            head_size=head_size,
            kv_num_heads=self.kv_num_heads,
            device=device
        )

        # 量化相关初始化
        self._quantization_cache = {}
        self._gptq_fusion = None
        self._is_quantized = None
        self._quant_group_size = self._get_quant_group_size()
        
        # 🔧 方案1+2: 初始化时转换数据类型
        self._converted_layers = set()  # 记录已转换的层

    def _convert_layer_dtypes_to_float16(self, layer, layer_idx: int):
        """
        将层的量化参数转换为float16，避免运行时转换
        """
        if layer_idx in self._converted_layers:
            return  # 已经转换过
        
        logger.info(f"Layer {layer_idx}: 转换量化参数为float16")
        
        # 转换QKV投影参数
        if hasattr(layer, 'c_attn'):
            qkv_layer = layer.c_attn
            logger.info(f"🔍 QKV层参数检查:")
            
            if hasattr(qkv_layer, 'scales'):
                logger.info(f"  scales: {qkv_layer.scales.shape}, dtype: {qkv_layer.scales.dtype}")
                if qkv_layer.scales.dtype != torch.float16:
                    logger.info(f"🔄 转换QKV scales: {qkv_layer.scales.dtype} -> float16")
                    qkv_layer.scales = qkv_layer.scales.to(torch.float16)
                    logger.info(f"✅ QKV scales转换后: {qkv_layer.scales.dtype}")
                else:
                    logger.info(f"✅ QKV scales已经是正确类型: {qkv_layer.scales.dtype}")
            
            if hasattr(qkv_layer, 'qweight'):
                logger.info(f"  qweight: {qkv_layer.qweight.shape}, dtype: {qkv_layer.qweight.dtype}")
                if qkv_layer.qweight.dtype != torch.uint32:
                    logger.info(f"🔄 转换QKV qweight: {qkv_layer.qweight.dtype} -> uint32")
                    qkv_layer.qweight = qkv_layer.qweight.to(torch.uint32)
                    logger.info(f"✅ QKV qweight转换后: {qkv_layer.qweight.dtype}")
                else:
                    logger.info(f"✅ QKV qweight已经是正确类型: {qkv_layer.qweight.dtype}")
            
            if hasattr(qkv_layer, 'qzeros'):
                logger.info(f"  qzeros: {qkv_layer.qzeros.shape}, dtype: {qkv_layer.qzeros.dtype}")
                if qkv_layer.qzeros.dtype != torch.uint32:
                    logger.info(f"🔄 转换QKV qzeros: {qkv_layer.qzeros.dtype} -> uint32")
                    qkv_layer.qzeros = qkv_layer.qzeros.to(torch.uint32)
                    logger.info(f"✅ QKV qzeros转换后: {qkv_layer.qzeros.dtype}")
                else:
                    logger.info(f"✅ QKV qzeros已经是正确类型: {qkv_layer.qzeros.dtype}")
        
        # 转换输出投影参数
        if hasattr(layer, 'c_proj'):
            proj_layer = layer.c_proj
            logger.info(f"🔍 输出投影层参数检查:")
            
            if hasattr(proj_layer, 'scales'):
                logger.info(f"  scales: {proj_layer.scales.shape}, dtype: {proj_layer.scales.dtype}")
                if proj_layer.scales.dtype != torch.float16:
                    logger.info(f"🔄 转换输出投影 scales: {proj_layer.scales.dtype} -> float16")
                    proj_layer.scales = proj_layer.scales.to(torch.float16)
                    logger.info(f"✅ 输出投影 scales转换后: {proj_layer.scales.dtype}")
                else:
                    logger.info(f"✅ 输出投影 scales已经是正确类型: {proj_layer.scales.dtype}")
            
            if hasattr(proj_layer, 'qweight'):
                logger.info(f"  qweight: {proj_layer.qweight.shape}, dtype: {proj_layer.qweight.dtype}")
                if proj_layer.qweight.dtype != torch.uint32:
                    logger.info(f"🔄 转换输出投影 qweight: {proj_layer.qweight.dtype} -> uint32")
                    proj_layer.qweight = proj_layer.qweight.to(torch.uint32)
                    logger.info(f"✅ 输出投影 qweight转换后: {proj_layer.qweight.dtype}")
                else:
                    logger.info(f"✅ 输出投影 qweight已经是正确类型: {proj_layer.qweight.dtype}")
            
            if hasattr(proj_layer, 'qzeros'):
                logger.info(f"  qzeros: {proj_layer.qzeros.shape}, dtype: {proj_layer.qzeros.dtype}")
                if proj_layer.qzeros.dtype != torch.uint32:
                    logger.info(f"🔄 转换输出投影 qzeros: {proj_layer.qzeros.dtype} -> uint32")
                    proj_layer.qzeros = proj_layer.qzeros.to(torch.uint32)
                    logger.info(f"✅ 输出投影 qzeros转换后: {proj_layer.qzeros.dtype}")
                else:
                    logger.info(f"✅ 输出投影 qzeros已经是正确类型: {proj_layer.qzeros.dtype}")
        
        # 转换MLP参数
        if hasattr(layer, 'mlp'):
            mlp_layer = layer.mlp
            logger.info(f"🔍 MLP层参数检查:")
            
            # 转换gate_proj
            if hasattr(mlp_layer, 'gate_proj'):
                gate_proj = mlp_layer.gate_proj
                if hasattr(gate_proj, 'scales'):
                    logger.info(f"  gate_proj scales: {gate_proj.scales.shape}, dtype: {gate_proj.scales.dtype}")
                    if gate_proj.scales.dtype != torch.float16:
                        logger.info(f"🔄 转换MLP gate_proj scales: {gate_proj.scales.dtype} -> float16")
                        gate_proj.scales = gate_proj.scales.to(torch.float16)
                        logger.info(f"✅ MLP gate_proj scales转换后: {gate_proj.scales.dtype}")
                    else:
                        logger.info(f"✅ MLP gate_proj scales已经是正确类型: {gate_proj.scales.dtype}")
                
                if hasattr(gate_proj, 'qweight'):
                    logger.info(f"  gate_proj qweight: {gate_proj.qweight.shape}, dtype: {gate_proj.qweight.dtype}")
                    if gate_proj.qweight.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP gate_proj qweight: {gate_proj.qweight.dtype} -> uint32")
                        gate_proj.qweight = gate_proj.qweight.to(torch.uint32)
                        logger.info(f"✅ MLP gate_proj qweight转换后: {gate_proj.qweight.dtype}")
                    else:
                        logger.info(f"✅ MLP gate_proj qweight已经是正确类型: {gate_proj.qweight.dtype}")
                
                if hasattr(gate_proj, 'qzeros'):
                    logger.info(f"  gate_proj qzeros: {gate_proj.qzeros.shape}, dtype: {gate_proj.qzeros.dtype}")
                    if gate_proj.qzeros.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP gate_proj qzeros: {gate_proj.qzeros.dtype} -> uint32")
                        gate_proj.qzeros = gate_proj.qzeros.to(torch.uint32)
                        logger.info(f"✅ MLP gate_proj qzeros转换后: {gate_proj.qzeros.dtype}")
                    else:
                        logger.info(f"✅ MLP gate_proj qzeros已经是正确类型: {gate_proj.qzeros.dtype}")
            
            # 转换up_proj
            if hasattr(mlp_layer, 'up_proj'):
                up_proj = mlp_layer.up_proj
                if hasattr(up_proj, 'scales'):
                    logger.info(f"  up_proj scales: {up_proj.scales.shape}, dtype: {up_proj.scales.dtype}")
                    if up_proj.scales.dtype != torch.float16:
                        logger.info(f"🔄 转换MLP up_proj scales: {up_proj.scales.dtype} -> float16")
                        up_proj.scales = up_proj.scales.to(torch.float16)
                        logger.info(f"✅ MLP up_proj scales转换后: {up_proj.scales.dtype}")
                    else:
                        logger.info(f"✅ MLP up_proj scales已经是正确类型: {up_proj.scales.dtype}")
                
                if hasattr(up_proj, 'qweight'):
                    logger.info(f"  up_proj qweight: {up_proj.qweight.shape}, dtype: {up_proj.qweight.dtype}")
                    if up_proj.qweight.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP up_proj qweight: {up_proj.qweight.dtype} -> uint32")
                        up_proj.qweight = up_proj.qweight.to(torch.uint32)
                        logger.info(f"✅ MLP up_proj qweight转换后: {up_proj.qweight.dtype}")
                    else:
                        logger.info(f"✅ MLP up_proj qweight已经是正确类型: {up_proj.qweight.dtype}")
                
                if hasattr(up_proj, 'qzeros'):
                    logger.info(f"  up_proj qzeros: {up_proj.qzeros.shape}, dtype: {up_proj.qzeros.dtype}")
                    if up_proj.qzeros.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP up_proj qzeros: {up_proj.qzeros.dtype} -> uint32")
                        up_proj.qzeros = up_proj.qzeros.to(torch.uint32)
                        logger.info(f"✅ MLP up_proj qzeros转换后: {up_proj.qzeros.dtype}")
                    else:
                        logger.info(f"✅ MLP up_proj qzeros已经是正确类型: {up_proj.qzeros.dtype}")
            
            # 转换down_proj
            if hasattr(mlp_layer, 'down_proj'):
                down_proj = mlp_layer.down_proj
                if hasattr(down_proj, 'scales'):
                    logger.info(f"  down_proj scales: {down_proj.scales.shape}, dtype: {down_proj.scales.dtype}")
                    if down_proj.scales.dtype != torch.float16:
                        logger.info(f"🔄 转换MLP down_proj scales: {down_proj.scales.dtype} -> float16")
                        down_proj.scales = down_proj.scales.to(torch.float16)
                        logger.info(f"✅ MLP down_proj scales转换后: {down_proj.scales.dtype}")
                    else:
                        logger.info(f"✅ MLP down_proj scales已经是正确类型: {down_proj.scales.dtype}")
                
                if hasattr(down_proj, 'qweight'):
                    logger.info(f"  down_proj qweight: {down_proj.qweight.shape}, dtype: {down_proj.qweight.dtype}")
                    if down_proj.qweight.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP down_proj qweight: {down_proj.qweight.dtype} -> uint32")
                        down_proj.qweight = down_proj.qweight.to(torch.uint32)
                        logger.info(f"✅ MLP down_proj qweight转换后: {down_proj.qweight.dtype}")
                    else:
                        logger.info(f"✅ MLP down_proj qweight已经是正确类型: {down_proj.qweight.dtype}")
                
                if hasattr(down_proj, 'qzeros'):
                    logger.info(f"  down_proj qzeros: {down_proj.qzeros.shape}, dtype: {down_proj.qzeros.dtype}")
                    if down_proj.qzeros.dtype != torch.uint32:
                        logger.info(f"🔄 转换MLP down_proj qzeros: {down_proj.qzeros.dtype} -> uint32")
                        down_proj.qzeros = down_proj.qzeros.to(torch.uint32)
                        logger.info(f"✅ MLP down_proj qzeros转换后: {down_proj.qzeros.dtype}")
                    else:
                        logger.info(f"✅ MLP down_proj qzeros已经是正确类型: {down_proj.qzeros.dtype}")
        
        # 标记为已转换
        self._converted_layers.add(layer_idx)
        logger.info(f"Layer {layer_idx}: 数据类型转换完成")

    def process_layer(self,
                      layer,
                      hidden_states: torch.Tensor,
                      cache_manager,
                      seq_ids: List[int],
                      context_lens: List[int],
                      token_positions: Optional[torch.Tensor] = None,
                      layer_idx: int = 0,
                      current_positions: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        start_time = time.time()

        # 🔧 方案1+2: 首次处理时转换数据类型
        self._convert_layer_dtypes_to_float16(layer, layer_idx)

        # 1. 检测量化状态
        if self._is_quantized is None:
            self._is_quantized = self._detect_qwen_quantized_model(layer)
            if self._is_quantized:
                self._gptq_fusion = GPTQCUDAFusion(groupsize=self._quant_group_size)
                logger.info(f"Layer {layer_idx}: 检测到量化模型，启用CUDA融合优化")

        # 记录LayerNorm前的时间
        norm_start = time.time()
        logger.info(f"hidden_states start shape {hidden_states.shape}")

        # 2. LayerNorm + 残差
        residual = hidden_states
        hidden_states = layer.ln_1(hidden_states)

        logger.info(f"hidden_states norm_fn shape {hidden_states.shape}")
        norm_time = time.time() - norm_start

        # 3. QKV计算 (优化版本)
        qkv_start = time.time()

        # 🔧 方案1+2: 确保input数据类型为float16
        if hidden_states.dtype == torch.bfloat16:
            logger.debug(f"转换hidden_states从{hidden_states.dtype}到float16")
            hidden_states = hidden_states.to(torch.float16)

        if self._is_quantized:
            q, k, v = self._optimized_quantized_qkv_proj(layer, hidden_states, layer_idx)
        else:
            # 标准QKV计算
            qkv = layer.attn.c_attn(hidden_states)
            logger.info(f"qkv   shape {qkv.shape}")
            hidden_size = qkv.shape[-1] // 3
            q, k, v = qkv.split(hidden_size, dim=-1)
            batch_size, seq_len, _ = hidden_states.shape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
            k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)

        qkv_time = time.time() - qkv_start

        # 4. 注意力计算
        attn_start = time.time()
        attn_output = self.attention(
            query=q.squeeze(2),  # [B, H, D]
            cache_manager=cache_manager,
            seq_ids=seq_ids,
            context_lens=context_lens,
            layer_idx=layer_idx,
            key=k.squeeze(2),  # [B, H, D]
            value=v.squeeze(2)  # [B, H, D]
        )
        logger.info(f"attn  attn_output shape {attn_output.shape}")
        attn_time = time.time() - attn_start

        # 5. 输出投影 (优化版本)
        proj_start = time.time()

        if self._is_quantized:
            attn_output = self._optimized_quantized_out_proj(layer, attn_output, layer_idx)
            batch_size, seq_len, _ = hidden_states.shape
            attn_output = attn_output.view(batch_size, seq_len, -1)
        else:
            attn_output = layer.attn.c_proj(attn_output.reshape(hidden_states.shape[0], -1)).unsqueeze(1)

        logger.info(f"proj_fn  attn_output shape {attn_output.shape}")
        hidden_states = residual + attn_output
        logger.info(f"proj  hidden_states shape {hidden_states.shape}")
        proj_time = time.time() - proj_start

        # 6. MLP + 残差
        mlp_start = time.time()
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        logger.info(f"mlp  hidden_states shape {hidden_states.shape}")
        mlp_time = time.time() - mlp_start

        # 记录总耗时
        total_time = time.time() - start_time
        if layer_idx == 0:
            logger.info(f"Layer {layer_idx}: 总处理耗时 {total_time * 1000:.2f}ms, "
                       f"分布: LN({norm_time * 1000:.2f}ms)+QKV({qkv_time * 1000:.2f}ms)+"
                       f"Attn({attn_time * 1000:.2f}ms)+Proj({proj_time * 1000:.2f}ms)+MLP({mlp_time * 1000:.2f}ms) | "
                       f"Quant: {self._is_quantized}")

        return hidden_states, (k.squeeze(2), v.squeeze(2))

    def _optimized_quantized_qkv_proj(self, layer, hidden_states: torch.Tensor, layer_idx: int):
        """
        优化的量化QKV投影 - 使用CUDA融合算子
        """
        c_attn = layer.attn.c_attn

        # 缓存量化参数
        cache_key = f"qkv_{id(c_attn)}"
        if cache_key not in self._quantization_cache:
            qkv_weight, qkv_scale, qkv_zero = self._get_quant_params(c_attn)
            self._quantization_cache[cache_key] = (qkv_weight, qkv_scale, qkv_zero)
        else:
            qkv_weight, qkv_scale, qkv_zero = self._quantization_cache[cache_key]

        # 重塑输入为 [M, K] 格式
        batch_size, seq_len, hidden_dim = hidden_states.shape
        input_2d = hidden_states.view(-1, hidden_dim)  # [B*S, D]
        
        # 🔧 调试：检查输入和量化参数类型
        logger.info(f"🔍 QKV投影参数类型检查:")
        logger.info(f"  input_2d: {input_2d.shape}, dtype: {input_2d.dtype}")
        logger.info(f"  qkv_weight: {qkv_weight.shape}, dtype: {qkv_weight.dtype}")
        logger.info(f"  qkv_scale: {qkv_scale.shape}, dtype: {qkv_scale.dtype}")
        logger.info(f"  qkv_zero: {qkv_zero.shape}, dtype: {qkv_zero.dtype}")
        
        # 调用CUDA融合算子
        result = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=input_2d,
            qweight=qkv_weight,
            qzeros=qkv_zero,
            scales=qkv_scale
        )
        
        # 重塑输出为 [B*S, output_dim]
        result = result.view(batch_size, seq_len, -1)
        
        # 分割QKV
        output_dim = result.shape[-1]
        if output_dim % 3 != 0:
            logger.error(f"QKV output dimension {output_dim} is not divisible by 3")
            raise ValueError(f"QKV output dimension {output_dim} is not divisible by 3")
        
        hidden_size = output_dim // 3
        q, k, v = result.split(hidden_size, dim=-1)
        
        # 重塑为 [B, H, S, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        
        return q, k, v

    def _optimized_quantized_out_proj(self, layer, attn_output: torch.Tensor, layer_idx: int):
        """
        优化的量化输出投影 - 使用CUDA融合算子
        """
        c_proj = layer.attn.c_proj

        # 缓存量化参数
        cache_key = f"out_{id(c_proj)}"
        if cache_key not in self._quantization_cache:
            out_weight, out_scale, out_zero = self._get_quant_params(c_proj)
            self._quantization_cache[cache_key] = (out_weight, out_scale, out_zero)
        else:
            out_weight, out_scale, out_zero = self._quantization_cache[cache_key]

        # 重塑注意力输出: [batch_size, num_heads, head_size] -> [batch_size, seq_len, hidden_size]
        batch_size, num_heads, head_size = attn_output.shape
        hidden_size = num_heads * head_size  # 4096 = 32 * 128
        
        # 重塑为 [batch_size, 1, hidden_size]
        attn_output_reshaped = attn_output.view(batch_size, 1, hidden_size)
        
        # 使用CUDA融合算子
        input_2d = attn_output_reshaped.view(-1, hidden_size)  # [B*S, D]
        
        result = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=input_2d,
            qweight=out_weight,
            qzeros=out_zero,
            scales=out_scale
        )
        
        # 重塑输出
        result = result.view(batch_size, 1, -1)
        return result

    def _get_quant_group_size(self) -> int:
        """获取量化组大小"""
        if hasattr(self.config, "quantization_config"):
            quant_config = self.config.quantization_config
            if isinstance(quant_config, dict) and "group_size" in quant_config:
                return quant_config["group_size"]
        if hasattr(self.config, "group_size"):
            return self.config.group_size
        return 128

    def _detect_qwen_quantized_model(self, layer) -> bool:
        """检测是否为量化模型"""
        if not hasattr(layer, "attn") or not hasattr(layer.attn, "c_attn"):
            return False

        c_attn = layer.attn.c_attn
        return (hasattr(c_attn, "qweight") or 
                (hasattr(c_attn, "weight") and c_attn.weight.dtype == torch.int8))

    def _get_quant_params(self, module):
        """获取量化参数"""
        if hasattr(module, "qweight"):
            # 🔧 确保参数类型正确
            qweight = module.qweight
            scales = module.scales
            qzeros = module.qzeros
            
            # 转换scales为float16
            if scales.dtype != torch.float16:
                logger.info(f"🔄 _get_quant_params转换scales: {scales.dtype} -> float16")
                scales = scales.to(torch.float16)
            
            # 转换qweight为uint32
            if qweight.dtype != torch.uint32:
                logger.info(f"🔄 _get_quant_params转换qweight: {qweight.dtype} -> uint32")
                qweight = qweight.to(torch.uint32)
            
            # 转换qzeros为uint32
            if qzeros.dtype != torch.uint32:
                logger.info(f"🔄 _get_quant_params转换qzeros: {qzeros.dtype} -> uint32")
                qzeros = qzeros.to(torch.uint32)
            
            return qweight, scales, qzeros
        
        if hasattr(module, "weight") and hasattr(module, "scales"):
            weight = module.weight
            scale = module.scales
            zero = getattr(module, "zeros", getattr(module, "qzeros", None))
            if zero is None:
                zero = torch.zeros_like(scale)
            
            # 转换scales为float16
            if scale.dtype != torch.float16:
                logger.info(f"🔄 _get_quant_params转换scales: {scale.dtype} -> float16")
                scale = scale.to(torch.float16)
            
            return weight, scale, zero

        raise AttributeError(f"无法获取量化参数: {module.__class__.__name__}")

    def clear_cache(self):
        """清理缓存"""
        self._quantization_cache.clear()
        self._is_quantized = None


# 添加别名以便导入
OptimizedQwenLayer = OptimizedQwenModelLayerAdapter