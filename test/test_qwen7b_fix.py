#!/usr/bin/env python3
"""
测试实际Qwen7B模型的QKV投影修复
"""
import torch
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志级别
logging.basicConfig(level=logging.INFO)

from core.layer.gptq import GPTQTritonFusion

def test_qwen7b_qkv_fix():
    """测试Qwen7B的QKV投影修复"""
    print("🧪 测试Qwen7B QKV投影修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # Qwen7B模型参数
    hidden_size = 4096
    qkv_output_dim = hidden_size * 3  # 12288
    
    print(f"Qwen7B模型参数:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  QKV输出维度: {qkv_output_dim}")
    
    # 模拟实际推理场景
    M, K = 1, hidden_size  # batch_size=1, input_dim=4096
    groupsize = 128
    
    # 创建测试数据
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    # 模拟实际的GPTQ参数格式 - 使用错误的维度来测试修复
    num_groups = K // groupsize
    qweight = torch.randint(0, 256, (512, qkv_output_dim), dtype=torch.int32, device='cuda')  # [512, 12288]
    qzeros = torch.randint(0, 16, (num_groups, 1536), dtype=torch.int32, device='cuda')  # [32, 1536]
    scales = torch.randn(num_groups, qkv_output_dim, dtype=torch.float16, device='cuda')  # [32, 12288]
    
    print(f"\n测试参数:")
    print(f"  输入形状: {input_tensor.shape}")
    print(f"  qweight形状: {qweight.shape}")
    print(f"  qzeros形状: {qzeros.shape}")
    print(f"  scales形状: {scales.shape}")
    
    # 创建GPTQ融合对象
    fusion = GPTQTritonFusion(groupsize=groupsize)
    
    try:
        # 测试基线实现
        print("\n🐌 测试基线实现...")
        baseline_result = fusion.baseline_gptq_gemm(
            input=input_tensor,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            groupsize=groupsize
        )
        print(f"✅ 基线实现成功: 输出形状 {baseline_result.shape}")
        
        # 检查输出维度
        output_dim = baseline_result.shape[-1]
        print(f"📊 实际输出维度: {output_dim}")
        
        # 验证输出维度
        if output_dim == qkv_output_dim:
            print(f"🎉 输出维度正确: {output_dim}")
            
            # 测试分割
            if output_dim % 3 == 0:
                hidden_size = output_dim // 3
                print(f"✅ 可以分割为3部分，每部分维度: {hidden_size}")
                
                # 测试分割
                q, k, v = baseline_result.split(hidden_size, dim=-1)
                print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
                
                # 验证分割后的维度
                if q.shape[-1] == hidden_size and k.shape[-1] == hidden_size and v.shape[-1] == hidden_size:
                    print(f"🎉 分割维度正确: 每个部分都是 {hidden_size} 维度")
                    return True
                else:
                    print(f"❌ 分割维度错误")
                    return False
            else:
                print(f"❌ 输出维度 {output_dim} 不能被3整除")
                return False
        else:
            print(f"❌ 输出维度错误: 期望 {qkv_output_dim}, 实际 {output_dim}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen7b_layer_integration():
    """测试Qwen7B层集成"""
    print("\n🧪 测试Qwen7B层集成...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    # 模拟层适配器的QKV投影
    from core.layer.optimized_qwen_layer import OptimizedQwenModelLayerAdapter
    
    # 创建模拟的层配置
    class MockConfig:
        def __init__(self):
            self.hidden_size = 4096
            self.num_heads = 32
            self.head_size = 128
            self.kv_num_heads = 32
    
    config = MockConfig()
    
    # 创建层适配器
    adapter = OptimizedQwenModelLayerAdapter(
        model_config=config,
        device='cuda',
        num_heads=32,
        head_size=128,
        kv_num_heads=32
    )
    
    # 模拟量化状态
    adapter._is_quantized = True
    adapter._gptq_fusion = GPTQTritonFusion(groupsize=128)
    
    # 模拟量化参数
    qkv_weight = torch.randint(0, 256, (512, 12288), dtype=torch.int32, device='cuda')
    qkv_scale = torch.randn(32, 12288, dtype=torch.float16, device='cuda')
    qkv_zero = torch.randint(0, 16, (32, 1536), dtype=torch.int32, device='cuda')
    
    # 缓存量化参数
    adapter._quantization_cache["qkv_test"] = (qkv_weight, qkv_scale, qkv_zero)
    
    # 模拟输入
    hidden_states = torch.randn(1, 1, 4096, dtype=torch.bfloat16, device='cuda')
    
    try:
        # 测试QKV投影
        print("测试QKV投影...")
        result = adapter._gptq_fusion.baseline_gptq_gemm(
            input=hidden_states.view(-1, 4096),
            qweight=qkv_weight,
            qzeros=qkv_zero,
            scales=qkv_scale,
            groupsize=128
        )
        
        print(f"QKV投影结果形状: {result.shape}")
        
        # 重塑输出
        result = result.view(1, 1, -1)
        print(f"重塑后形状: {result.shape}")
        
        # 检查输出维度
        output_dim = result.shape[-1]
        if output_dim % 3 == 0:
            hidden_size = output_dim // 3
            print(f"✅ 输出维度 {output_dim} 可以被3整除，每部分维度: {hidden_size}")
            
            # 测试分割
            q, k, v = result.split(hidden_size, dim=-1)
            print(f"✅ 分割成功: Q{q.shape}, K{k.shape}, V{v.shape}")
            return True
        else:
            print(f"❌ 输出维度 {output_dim} 不能被3整除")
            return False
            
    except Exception as e:
        print(f"❌ 层集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Qwen7B QKV投影修复测试开始...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，所有测试跳过")
        return
    
    tests = [
        ("Qwen7B QKV投影修复测试", test_qwen7b_qkv_fix),
        ("Qwen7B层集成测试", test_qwen7b_layer_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有Qwen7B修复测试通过！")
    else:
        print("⚠️  部分Qwen7B修复测试失败")

if __name__ == "__main__":
    main()
