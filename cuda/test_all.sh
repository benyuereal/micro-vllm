#!/bin/bash
"""
CUDA内核一键测试 - 目标0.10ms
"""

echo "🚀 CUDA内核一键测试开始..."
echo "=========================================="
echo "🎯 目标性能: 0.10ms"
echo "=========================================="

# 步骤1: 检查环境
echo "🔍 步骤1: 检查CUDA环境"
echo "=========================================="
if ! python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" 2>/dev/null; then
    echo "❌ CUDA环境检查失败"
    exit 1
fi

# 步骤2: 编译内核
echo "🔨 步骤2: 编译CUDA内核"
echo "=========================================="
if ! python compile.py; then
    echo "❌ CUDA内核编译失败"
    exit 1
fi

# 步骤3: 测试内核
echo "⚡ 步骤3: 测试CUDA内核性能"
echo "=========================================="
if ! python test.py; then
    echo "❌ CUDA内核测试失败"
    exit 1
fi

echo "=========================================="
echo "🎉 CUDA内核测试完成!"
echo "🎯 目标性能: 0.10ms"
echo "=========================================="
