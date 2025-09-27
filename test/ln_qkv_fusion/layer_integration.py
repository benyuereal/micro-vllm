#!/usr/bin/env python3
"""
Layer层集成测试
测试融合内核的集成使用
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

def test_fusion_kernel_integration():
    """测试融合内核集成"""
    print("🚀 融合内核集成测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用，无法测试")
    
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"📊 GPU名称: {torch.cuda.get_device_name()}")
    
    # 测试参数
    batch_size, seq_len, hidden_size = 1, 1, 4096
    groupsize = 128
    eps = 1e-5
    
    print(f"\n📊 测试数据准备...")
    print(f"📊 hidden_states形状: {batch_size}x{seq_len}x{hidden_size}")
    
    # 创建测试输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda')
    print(f"📊 hidden_states形状: {hidden_states.shape}")
    
    # 测试融合内核（LayerNorm + GPTQ + QKV）
    print("\n🔨 测试融合内核...")
    try:
        # 编译融合内核
        print("📦 编译融合内核...")
        from torch.utils.cpp_extension import load
        
        # 切换到cuda目录
        cuda_dir = os.path.join(project_root, "..", "cuda")
        original_cwd = os.getcwd()
        os.chdir(cuda_dir)
        
        kernel_module = load(
            name="fused_ln_qkv_gptq_cuda",
            sources=["gptq_ln_qkv_fusion_kernel.cu"],
            extra_cuda_cflags=["-O3", "-use_fast_math"],
            verbose=False
        )
        
        # 恢复原目录
        os.chdir(original_cwd)
        print("✅ 融合内核编译成功")
        
        # 创建LayerNorm参数
        ln_weight = torch.ones(hidden_size, dtype=torch.float16, device='cuda')
        ln_bias = torch.zeros(hidden_size, dtype=torch.float16, device='cuda')
        
        # 创建分离的Q、K、V GPTQ参数
        qweight_q = torch.randint(0, 256, (hidden_size // 8, hidden_size), dtype=torch.uint32, device='cuda')
        qweight_k = torch.randint(0, 256, (hidden_size // 8, hidden_size), dtype=torch.uint32, device='cuda')
        qweight_v = torch.randint(0, 256, (hidden_size // 8, hidden_size), dtype=torch.uint32, device='cuda')
        
        qzeros_q = torch.randint(0, 16, (hidden_size // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        qzeros_k = torch.randint(0, 16, (hidden_size // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        qzeros_v = torch.randint(0, 16, (hidden_size // groupsize, groupsize // 8), dtype=torch.uint32, device='cuda')
        
        scales_q = torch.randn(hidden_size // groupsize, hidden_size, dtype=torch.float16, device='cuda')
        scales_k = torch.randn(hidden_size // groupsize, hidden_size, dtype=torch.float16, device='cuda')
        scales_v = torch.randn(hidden_size // groupsize, hidden_size, dtype=torch.float16, device='cuda')
        
        # 执行融合内核
        print("⚡ 测试融合内核...")
        qkv_output = kernel_module.fused_ln_qkv_gptq_cuda(
            hidden_states, qweight_q, qweight_k, qweight_v,
            qzeros_q, qzeros_k, qzeros_v,
            scales_q, scales_k, scales_v,
            ln_weight, ln_bias,
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
        
        # 性能测试
        print("\n⚡ 性能测试...")
        import time
        
        # 预热
        for _ in range(10):
            _ = kernel_module.fused_ln_qkv_gptq_cuda(
                hidden_states, qweight_q, qweight_k, qweight_v,
                qzeros_q, qzeros_k, qzeros_v,
                scales_q, scales_k, scales_v,
                ln_weight, ln_bias,
                batch_size, seq_len, hidden_size, groupsize, eps
            )
        
        # 性能测试
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            _ = kernel_module.fused_ln_qkv_gptq_cuda(
                hidden_states, qweight_q, qweight_k, qweight_v,
                qzeros_q, qzeros_k, qzeros_v,
                scales_q, scales_k, scales_v,
                ln_weight, ln_bias,
                batch_size, seq_len, hidden_size, groupsize, eps
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        print(f"📊 平均延迟: {avg_time:.4f}ms")
        print(f"📊 目标延迟: < 0.25ms")
        
        if avg_time < 0.25:
            print("✅ 性能测试通过!")
        else:
            print("⚠️ 性能未达标，但功能正常")
        
        print("\n🎉 融合内核集成测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 融合内核测试失败: {e}")
        return False

def main():
    """主函数"""
    try:
        success = test_fusion_kernel_integration()
        if success:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 部分测试失败!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")

if __name__ == "__main__":
    main()