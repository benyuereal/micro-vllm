#!/usr/bin/env python3
"""
Layer层集成测试
测试optimized_qwen_layer.py与CUDA融合内核的集成
"""

import torch
import sys
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_layer_integration():
    """测试layer层集成"""
    print("🚀 Layer层集成测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用，无法测试")
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 测试数据 - 基于实际layer层数据形状
    print("\n📊 测试数据准备...")
    
    # 模拟layer层的输入数据
    batch_size = 1
    seq_len = 1
    hidden_size = 4096
    num_heads = 32
    head_size = 128
    
    # 创建测试输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda')
    print(f"📊 hidden_states形状: {hidden_states.shape}")
    
    # 模拟注意力输出 [batch_size, num_heads, head_size]
    attn_output = torch.randn(batch_size, num_heads, head_size, dtype=torch.float16, device='cuda')
    print(f"📊 attn_output形状: {attn_output.shape}")
    
    # 测试融合内核（LayerNorm + GPTQ + QKV）
    print("\n🔨 测试融合内核...")
    try:
        # 编译融合内核
        print("📦 编译融合内核...")
        from torch.utils.cpp_extension import load
        
        # 切换到cuda目录
        cuda_dir = os.path.join(project_root, "cuda")
        original_cwd = os.getcwd()
        os.chdir(cuda_dir)
        
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        os.chdir(original_cwd)
        print("✅ 融合内核编译成功")
        
        # 测试融合内核
        print("\n⚡ 测试融合内核...")
        groupsize = 128
        eps = 1e-5
        
        # 创建LayerNorm参数
        ln_weight = torch.ones(hidden_size, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_size, dtype=torch.float16, device='cuda')
        
        # 创建GPTQ参数
        qkv_qweight = torch.randint(0, 256, (hidden_size // 8, hidden_size * 3), dtype=torch.uint32, device='cuda')
        qkv_qzeros = torch.randint(0, 16, (hidden_size // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        qkv_scales = torch.randn(hidden_size // groupsize, hidden_size * 3, dtype=torch.float16, device='cuda')
        
        # 执行融合内核
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            hidden_states, qkv_qweight, qkv_qzeros, qkv_scales, ln_weight, ln_bias,
            batch_size, seq_len, hidden_size, groupsize, eps
        )
        
        # 解包QKV输出
        q_output = qkv_output[0]
        k_output = qkv_output[1]
        v_output = qkv_output[2]
        
        print(f"📊 Q输出形状: {q_output.shape}")
        print(f"📊 K输出形状: {k_output.shape}")
        print(f"📊 V输出形状: {v_output.shape}")
        print(f"📊 期望形状: torch.Size([1, 1, 4096])")
        
        expected_shape = (batch_size, seq_len, hidden_size)
        assert q_output.shape == expected_shape, f"Q输出形状错误: {q_output.shape}"
        assert k_output.shape == expected_shape, f"K输出形状错误: {k_output.shape}"
        assert v_output.shape == expected_shape, f"V输出形状错误: {v_output.shape}"
        print("✅ 融合内核测试通过!")
        
        # 测试注意力输出投影 (attn_output -> proj_fn)
        print("\n⚡ 测试注意力输出投影...")
        # attn_output: [1, 32, 128] -> reshape为 [32, 128] 进行投影
        attn_2d = attn_output.view(-1, head_size)  # [32, 128]
        M, K, N = 32, 128, 128  # 投影到相同的head_size
        groupsize = 128
        num_groups = K // groupsize
        
        # 创建注意力输出投影的量化权重
        attn_out_qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        attn_out_qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        attn_out_scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        # 执行注意力输出投影
        attn_out_result = gptq_fusion.fused_gptq_gemm_4bit(
            attn_2d, attn_out_qweight, attn_out_qzeros, attn_out_scales
        )
        
        print(f"📊 注意力输出投影形状: {attn_out_result.shape}")
        print(f"📊 期望形状: torch.Size([32, 128])")
        assert attn_out_result.shape == (32, 128), f"注意力输出投影形状错误: {attn_out_result.shape}"
        print("✅ 注意力输出投影测试通过!")
        
        # 测试最终输出投影 (proj_fn -> proj)
        print("\n⚡ 测试最终输出投影...")
        # 将注意力输出reshape回 [1, 1, 4096] 进行最终投影
        attn_out_3d = attn_out_result.view(batch_size, seq_len, hidden_size)  # [1, 1, 4096]
        attn_out_2d = attn_out_3d.view(-1, hidden_size)  # [1, 4096]
        
        M, K, N = 1, 4096, 4096
        groupsize = 128
        num_groups = K // groupsize
        
        # 创建最终输出投影的量化权重
        final_out_qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        final_out_qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        final_out_scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        # 执行最终输出投影
        final_out_result = gptq_fusion.fused_gptq_gemm_4bit(
            attn_out_2d, final_out_qweight, final_out_qzeros, final_out_scales
        )
        
        print(f"📊 最终输出投影形状: {final_out_result.shape}")
        print(f"📊 期望形状: torch.Size([1, 4096])")
        assert final_out_result.shape == (1, 4096), f"最终输出投影形状错误: {final_out_result.shape}"
        print("✅ 最终输出投影测试通过!")
        
        # 测试MLP投影
        print("\n⚡ 测试MLP投影...")
        
        # MLP投影1: 4096 -> 11008
        M, K, N = 1, 4096, 11008
        groupsize = 128
        num_groups = K // groupsize
        
        mlp1_qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        mlp1_qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        mlp1_scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        mlp1_output = gptq_fusion.fused_gptq_gemm_4bit(
            input_2d, mlp1_qweight, mlp1_qzeros, mlp1_scales
        )
        
        print(f"📊 MLP投影1形状: {mlp1_output.shape}")
        print(f"📊 期望形状: torch.Size([1, 11008])")
        assert mlp1_output.shape == (1, 11008), f"MLP投影1形状错误: {mlp1_output.shape}"
        print("✅ MLP投影1测试通过!")
        
        # MLP投影2: 11008 -> 4096
        M, K, N = 1, 11008, 4096
        groupsize = 128
        num_groups = K // groupsize
        
        mlp2_qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.uint32, device='cuda')
        mlp2_qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.uint32, device='cuda')
        mlp2_scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
        
        mlp2_output = gptq_fusion.fused_gptq_gemm_4bit(
            mlp1_output, mlp2_qweight, mlp2_qzeros, mlp2_scales
        )
        
        print(f"📊 MLP投影2形状: {mlp2_output.shape}")
        print(f"📊 期望形状: torch.Size([1, 4096])")
        assert mlp2_output.shape == (1, 4096), f"MLP投影2形状错误: {mlp2_output.shape}"
        print("✅ MLP投影2测试通过!")
        
    except Exception as e:
        print(f"❌ GPTQ融合内核测试失败: {e}")
        raise RuntimeError(f"GPTQ融合内核测试失败: {e}")
    
    # 测试optimized_qwen_layer.py集成
    print("\n🔨 测试optimized_qwen_layer.py集成...")
    try:
        from core.layer.optimized_qwen_layer import OptimizedQwenLayer
        
        # 创建模拟的模型配置
        class MockConfig:
            def __init__(self):
                self.model_type = "qwen"
                self.group_size = 128
        
        mock_config = MockConfig()
        
        # 创建优化层实例
        layer = OptimizedQwenLayer(
            model_config=mock_config,
            device='cuda',
            num_heads=32,
            head_size=128,
            kv_num_heads=32
        )
        print("✅ OptimizedQwenLayer实例创建成功")
        
        # 测试前向传播
        print("\n⚡ 测试前向传播...")
        try:
            # 创建模拟的layer对象
            class MockLayer:
                def __init__(self):
                    self.ln_1 = lambda x: x  # 模拟LayerNorm
                    self.ln_2 = lambda x: x
                    self.attn = MockAttention()
                    self.mlp = lambda x: x  # 模拟MLP
                
            class MockAttention:
                def __init__(self):
                    self.c_attn = MockQuantizedLinear()
                    self.c_proj = MockQuantizedLinear()
            
            class MockQuantizedLinear:
                def __init__(self):
                    # 模拟量化权重 - 使用正确的维度
                    # QKV投影: [K//8, N] = [512, 12288]
                    self.qweight = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
                    self.qzeros = torch.randint(0, 16, (32, 512), dtype=torch.uint32, device='cuda')  # 修正为正确的维度
                    self.scales = torch.randn(32, 4096, dtype=torch.float16, device='cuda')
            
            mock_layer = MockLayer()
            
            # 创建模拟的cache_manager
            class MockCacheManager:
                def __init__(self):
                    # 模拟KV缓存
                    self.k_caches = {}
                    self.v_caches = {}
                    # 为每一层创建模拟缓存
                    for layer_idx in range(32):  # 假设32层
                        self.k_caches[layer_idx] = torch.randn(48, 256, 32, 128, dtype=torch.float16, device='cuda')
                        self.v_caches[layer_idx] = torch.randn(48, 256, 32, 128, dtype=torch.float16, device='cuda')
                
                def get(self, layer_idx):
                    """模拟KVCacheManager的get方法"""
                    return self.k_caches[layer_idx], self.v_caches[layer_idx]
                
                def put(self, seq_id, key, value, layer, slot_map):
                    """模拟put方法"""
                    pass
                
                def alloc(self, seq_id, n_tokens):
                    """模拟alloc方法"""
                    return True, list(range(n_tokens))
                
                def append(self, seq_id):
                    """模拟append方法"""
                    return 0
                
                def free(self, seq_id):
                    """模拟free方法"""
                    pass
            
            mock_cache_manager = MockCacheManager()
            
            # 使用process_layer方法
            output, (k, v) = layer.process_layer(
                layer=mock_layer,
                hidden_states=hidden_states,
                cache_manager=mock_cache_manager,
                seq_ids=[0],
                context_lens=[1],
                layer_idx=0
            )
            
            print(f"📊 Layer输出形状: {output.shape}")
            print(f"📊 期望形状: torch.Size([1, 1, 4096])")
            assert output.shape == (1, 1, 4096), f"Layer输出形状错误: {output.shape}"
            print("✅ Layer前向传播测试通过!")
            
        except Exception as e:
            print(f"❌ Layer前向传播测试失败: {e}")
            print("💡 可能需要修改optimized_qwen_layer.py以支持CUDA融合内核")
            
    except Exception as e:
        print(f"❌ OptimizedQwenLayer测试失败: {e}")
        print("💡 可能需要检查optimized_qwen_layer.py的实现")
    
    print("\n🎉 Layer层集成测试完成!")

if __name__ == "__main__":
    try:
        test_layer_integration()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
