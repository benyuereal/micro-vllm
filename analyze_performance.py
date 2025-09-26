#!/usr/bin/env python3
"""
分析GPTQ性能问题
"""
import torch
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer.gptq import GPTQTritonFusion

def analyze_performance():
    """分析GPTQ性能问题"""
    print("🔍 GPTQ性能分析...")
    
    # 测试不同大小的矩阵
    test_cases = [
        (4, 128, 32, "超小矩阵"),
        (32, 512, 128, "小矩阵"),
        (128, 1024, 512, "中等矩阵"),
        (512, 2048, 1024, "大矩阵"),
        (1024, 4096, 2048, "超大矩阵"),
    ]
    
    fusion = GPTQTritonFusion(groupsize=128)
    
    print("矩阵大小 | 融合算子(ms) | 基线实现(ms) | 加速比 | 状态")
    print("-" * 60)
    
    for M, K, N, desc in test_cases:
        try:
            # 创建测试数据
            input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
            qweight = torch.randint(0, 256, (N, K // 8), dtype=torch.int32, device='cuda')
            num_groups = K // 128
            scales = torch.randn(num_groups, K, dtype=torch.float16, device='cuda')
            qzeros = torch.randint(0, 16, (num_groups, K // 8), dtype=torch.int32, device='cuda')
            
            # 预热
            for _ in range(3):
                _ = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
            torch.cuda.synchronize()
            
            # 测试融合算子
            fusion_times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.time()
                _ = fusion.fused_gptq_gemm_4bit(input_tensor, qweight, qzeros, scales)
                torch.cuda.synchronize()
                fusion_times.append(time.time() - start)
            
            # 测试基线实现
            baseline_times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.time()
                _ = fusion.baseline_gptq_gemm(input_tensor, qweight, qzeros, scales, 128)
                torch.cuda.synchronize()
                baseline_times.append(time.time() - start)
            
            avg_fusion = sum(fusion_times) / len(fusion_times) * 1000
            avg_baseline = sum(baseline_times) / len(baseline_times) * 1000
            speedup = avg_baseline / avg_fusion
            
            status = "✅ 更快" if speedup > 1.0 else "❌ 更慢"
            
            print(f"{desc:8} | {avg_fusion:8.2f} | {avg_baseline:8.2f} | {speedup:6.2f} | {status}")
            
        except Exception as e:
            print(f"{desc:8} | 错误: {str(e)[:20]}...")
    
    print("\n📊 分析结论:")
    print("1. 小矩阵: Triton内核启动开销 > 计算收益")
    print("2. 大矩阵: GPU并行优势 > 内核启动开销")
    print("3. 建议: 在实际推理中使用大batch size")
    print("4. 优化方向: 减少内核启动开销，优化内存访问模式")

def suggest_optimizations():
    """提供优化建议"""
    print("\n🚀 GPTQ优化建议:")
    print("1. 使用更大的batch size (M维度)")
    print("2. 预热Triton内核，避免重复编译")
    print("3. 优化Triton内核的分块大小")
    print("4. 考虑使用CUDA kernels替代Triton")
    print("5. 实现内核缓存机制")
    
    print("\n💡 实际应用建议:")
    print("- 在推理时使用较大的batch size")
    print("- 预热阶段运行几次GPTQ融合算子")
    print("- 监控实际工作负载的矩阵大小")
    print("- 考虑动态选择融合算子或基线实现")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        sys.exit(1)
    
    analyze_performance()
    suggest_optimizations()
