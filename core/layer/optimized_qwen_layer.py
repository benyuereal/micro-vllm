# File: optimized_qwen_layer.py
import torch
from typing import Tuple, List, Optional
from core.paged_attention import PagedAttention
import logging
import time
from .gptq import GPTQCUDAFusion
import os

logger = logging.getLogger(__name__)


class OptimizedQwenModelLayerAdapter:
    """
    优化的Qwen7B INT4量化模型适配器
    专门用于Qwen7B INT4量化模型，强制使用LayerNorm+QKV融合内核
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
        
        # 融合内核初始化 - 必须成功
        self._fusion_kernel = None
        self._init_fusion_kernel()
        if self._fusion_kernel is None:
            raise RuntimeError("❌ 融合内核初始化失败，无法继续")

    def _init_fusion_kernel(self):
        """初始化融合内核 - 必须成功"""
        from torch.utils.cpp_extension import load
        
        # 获取CUDA内核文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_dir = os.path.join(current_dir, '..', '..', 'cuda')
        kernel_file = os.path.join(cuda_dir, 'gptq_ln_qkv_fusion_kernel.cu')
        
        if not os.path.exists(kernel_file):
            raise FileNotFoundError(f"融合内核文件不存在: {kernel_file}")
        
        # 编译融合内核
        self._fusion_kernel = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=[kernel_file],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        logger.info("✅ 融合内核加载成功")

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
        # 只在第一层打印形状信息
        if layer_idx == 0:
            logger.info(f"hidden_states start shape {hidden_states.shape}")

        # 2. 残差连接准备
        residual = hidden_states

        # 3. 融合的LayerNorm + QKV计算
        qkv_start = time.time()

        # 🔧 优化：只在需要时转换数据类型，避免每层都转换
        if hidden_states.dtype != torch.float16:
            if layer_idx == 0:
                logger.info(f"🔄 转换hidden_states从{hidden_states.dtype}到float16")
            hidden_states = hidden_states.to(torch.float16)
        elif layer_idx == 0:
            logger.info(f"✅ hidden_states已经是正确类型: {hidden_states.dtype}")

        # 强制使用融合内核：LayerNorm + QKV投影
        q, k, v = self._fused_ln_qkv_proj(layer, hidden_states, layer_idx)
        if layer_idx == 0:
            logger.info(f"🚀 使用融合内核进行LayerNorm+QKV投影")

        qkv_time = time.time() - qkv_start
        norm_time = 0  # 融合内核中LayerNorm时间包含在QKV时间中

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
        if layer_idx == 0:
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

        if layer_idx == 0:
            logger.info(f"proj_fn  attn_output shape {attn_output.shape}")
        hidden_states = residual + attn_output
        if layer_idx == 0:
            logger.info(f"proj  hidden_states shape {hidden_states.shape}")
        proj_time = time.time() - proj_start

        # 6. MLP + 残差
        mlp_start = time.time()
        residual = hidden_states
        hidden_states = layer.ln_2(hidden_states)
        
        # 🔧 优化：减少MLP数据类型转换开销
        if hidden_states.dtype != torch.bfloat16:
            if layer_idx == 0:
                logger.info(f"🔄 转换MLP输入: {hidden_states.dtype} -> bfloat16")
            hidden_states = hidden_states.to(torch.bfloat16)
        
        hidden_states = layer.mlp(hidden_states)
        
        # 🔧 优化：只在需要时转换回float16
        if hidden_states.dtype != torch.float16:
            if layer_idx == 0:
                logger.info(f"🔄 转换MLP输出: {hidden_states.dtype} -> float16")
            hidden_states = hidden_states.to(torch.float16)
        
        hidden_states = residual + hidden_states
        if layer_idx == 0:
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


    def _fused_ln_qkv_proj(self, layer, hidden_states: torch.Tensor, layer_idx: int):
        """
        融合的LayerNorm + QKV投影 - 使用CUDA融合内核
        """
        
        c_attn = layer.attn.c_attn
        
        # 获取量化参数
        cache_key = f"qkv_{id(c_attn)}"
        if cache_key not in self._quantization_cache:
            qkv_weight, qkv_scale, qkv_zero = self._get_quant_params(c_attn)
            self._quantization_cache[cache_key] = (qkv_weight, qkv_scale, qkv_zero)
        else:
            qkv_weight, qkv_scale, qkv_zero = self._quantization_cache[cache_key]
        
        # 分离Q、K、V权重
        hidden_dim = hidden_states.shape[-1]
        output_dim = qkv_weight.shape[1]  # 应该是 hidden_dim * 3
        if output_dim % 3 != 0:
            logger.error(f"QKV output dimension {output_dim} is not divisible by 3")
            raise ValueError(f"QKV output dimension {output_dim} is not divisible by 3")
        
        qkv_hidden_dim = output_dim // 3
        
        # 分离权重
        qweight_q = qkv_weight[:, :qkv_hidden_dim]
        qweight_k = qkv_weight[:, qkv_hidden_dim:2*qkv_hidden_dim]
        qweight_v = qkv_weight[:, 2*qkv_hidden_dim:]
        
        # 分离scales和zeros
        num_groups = qkv_scale.shape[0]
        qscales_q = qkv_scale[:, :qkv_hidden_dim]
        qscales_k = qkv_scale[:, qkv_hidden_dim:2*qkv_hidden_dim]
        qscales_v = qkv_scale[:, 2*qkv_hidden_dim:]
        
        qzeros_q = qkv_zero
        qzeros_k = qkv_zero  # 通常Q、K、V使用相同的zeros
        qzeros_v = qkv_zero
        
        # 获取RMSNorm参数 (Qwen使用RMSNorm，没有bias)
        rms_weight = layer.ln_1.weight.to(torch.float16)
        
        # 调用融合内核
        batch_size, seq_len, _ = hidden_states.shape
        groupsize = self._quant_group_size
        eps = 1e-5
        
        qkv_output = self._fusion_kernel.fused_ln_qkv_gptq_cuda(
            hidden_states, qweight_q, qweight_k, qweight_v,
            qzeros_q, qzeros_k, qzeros_v,
            qscales_q, qscales_k, qscales_v,
            rms_weight,
            batch_size, seq_len, qkv_hidden_dim, groupsize, eps
        )
        
        # 解包QKV输出
        q_output = qkv_output[0]
        k_output = qkv_output[1]
        v_output = qkv_output[2]
        
        # 重塑为注意力格式
        q = q_output.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k_output.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        v = v_output.view(batch_size, seq_len, self.kv_num_heads, self.head_size).permute(0, 2, 1, 3)
        
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

        # 🔧 优化：避免不必要的张量重塑
        batch_size, num_heads, head_size = attn_output.shape
        hidden_size = num_heads * head_size  # 4096 = 32 * 128
        
        # 🔧 优化：直接重塑为2D，避免中间变量
        input_2d = attn_output.view(batch_size, hidden_size)  # [B, D]
        
        # 🔧 优化：直接调用CUDA融合算子
        result = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=input_2d,
            qweight=out_weight,
            qzeros=out_zero,
            scales=out_scale
        )
        
        # 🔧 优化：直接重塑为3D输出
        return result.view(batch_size, 1, -1)

    def _get_quant_group_size(self) -> int:
        """获取量化组大小"""
        if hasattr(self.config, "quantization_config"):
            quant_config = self.config.quantization_config
            if isinstance(quant_config, dict) and "group_size" in quant_config:
                return quant_config["group_size"]
        if hasattr(self.config, "group_size"):
            return self.config.group_size
        return 128

    def _optimized_quantized_mlp(self, layer, hidden_states: torch.Tensor, layer_idx: int):
        """
        优化的量化MLP计算 - 使用CUDA融合算子
        MLP包含三个投影层：gate_proj, up_proj, down_proj
        """
        mlp_layer = layer.mlp
        
        # 1. Gate投影 (gate_proj)
        gate_proj = mlp_layer.gate_proj
        gate_cache_key = f"gate_{id(gate_proj)}"
        if gate_cache_key not in self._quantization_cache:
            gate_weight, gate_scale, gate_zero = self._get_quant_params(gate_proj)
            self._quantization_cache[gate_cache_key] = (gate_weight, gate_scale, gate_zero)
        else:
            gate_weight, gate_scale, gate_zero = self._quantization_cache[gate_cache_key]
        
        # 2. Up投影 (up_proj)
        up_proj = mlp_layer.up_proj
        up_cache_key = f"up_{id(up_proj)}"
        if up_cache_key not in self._quantization_cache:
            up_weight, up_scale, up_zero = self._get_quant_params(up_proj)
            self._quantization_cache[up_cache_key] = (up_weight, up_scale, up_zero)
        else:
            up_weight, up_scale, up_zero = self._quantization_cache[up_cache_key]
        
        # 3. Down投影 (down_proj)
        down_proj = mlp_layer.down_proj
        down_cache_key = f"down_{id(down_proj)}"
        if down_cache_key not in self._quantization_cache:
            down_weight, down_scale, down_zero = self._get_quant_params(down_proj)
            self._quantization_cache[down_cache_key] = (down_weight, down_scale, down_zero)
        else:
            down_weight, down_scale, down_zero = self._quantization_cache[down_cache_key]
        
        # 重塑输入为 [M, K] 格式
        batch_size, seq_len, hidden_dim = hidden_states.shape
        input_2d = hidden_states.view(-1, hidden_dim)  # [B*S, D]
        
        # 计算gate和up投影
        gate_output = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=input_2d,
            qweight=gate_weight,
            qzeros=gate_zero,
            scales=gate_scale
        )
        
        up_output = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=input_2d,
            qweight=up_weight,
            qzeros=up_zero,
            scales=up_scale
        )
        
        # 应用SiLU激活函数
        gate_output = torch.nn.functional.silu(gate_output)
        
        # 计算down投影
        down_input = gate_output * up_output
        down_output = self._gptq_fusion.fused_gptq_gemm_4bit(
            input=down_input,
            qweight=down_weight,
            qzeros=down_zero,
            scales=down_scale
        )
        
        # 重塑输出为 [B, S, D]
        result = down_output.view(batch_size, seq_len, -1)
        
        return result

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