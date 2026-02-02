#!/usr/bin/env python3
"""
RMSNorm 性能基准测试（专注速度对比）
- 仅测试前向推理性能
- 对比 Triton 实现 vs PyTorch 原生 RMSNorm
- 无额外依赖（不使用 pandas）
"""
import torch
import time
import sys
sys.path.insert(0, '.')  # 确保能导入同目录的 rmsnorm.py

from rmsnorm import rms_norm  # 引用您已有的实现


import torch
import torch.nn.functional as F

def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch 2.5+ 原生 RMSNorm"""
    return F.rms_norm(x, weight.shape, weight, eps)



def benchmark(func, x, weight, eps, num_warmup=50, num_iter=500):
    """精确计时（CUDA Event）"""
    # 预热
    for _ in range(num_warmup):
        _ = func(x, weight, eps)
    torch.cuda.synchronize()
    
    # 正式计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iter):
        _ = func(x, weight, eps)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iter


def main():
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"测试配置: FP16 输入/输出 | eps=1e-6 | iterations=500")
    print("=" * 80)
    print(f"{'Shape (B, H)':<20} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    configs = [
        (1, 512),
        (1, 1024),
        (1, 4096),
        (1, 8192),
        (8, 4096),
        (32, 4096),
        (128, 4096),
        (16, 12288),
    ]
    
    total_triton = 0.0
    total_torch = 0.0
    count = 0
    
    for batch_size, hidden_size in configs:
        # 创建 FP16 输入
        x = torch.randn((batch_size, hidden_size), device='cuda', dtype=torch.float16)
        weight = torch.randn((hidden_size,), device='cuda', dtype=torch.float16)
        eps = 1e-6
        
        # 测速
        triton_ms = benchmark(rms_norm, x, weight, eps)
        torch_ms = benchmark(rms_norm_torch, x, weight, eps)
        speedup = torch_ms / triton_ms
        
        # 累加用于平均
        total_triton += triton_ms
        total_torch += torch_ms
        count += 1
        
        # 输出
        shape_str = f"({batch_size}, {hidden_size})"
        print(f"{shape_str:<20} {triton_ms:<15.4f} {torch_ms:<15.4f} {speedup:<10.2f}x")
    
    # 汇总
    avg_speedup = total_torch / total_triton
    print("-" * 80)
    print(f"{'Average':<20} {'-':<15} {'-':<15} {avg_speedup:<10.2f}x")
    print("=" * 80)
    print(f"✅ Triton RMSNorm 平均加速 {avg_speedup:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()