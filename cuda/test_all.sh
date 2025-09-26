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

# 步骤2: 编译当前版本内核
echo "🔨 步骤2: 编译当前版本CUDA内核"
echo "=========================================="
if ! python compile.py; then
    echo "❌ 当前版本CUDA内核编译失败"
    exit 1
fi

# 步骤3: 测试当前版本
echo "⚡ 步骤3: 测试当前版本CUDA内核性能"
echo "=========================================="
if ! python test.py; then
    echo "❌ 当前版本CUDA内核测试失败"
    exit 1
fi

# 步骤4: 编译vLLM版本内核
echo "🔨 步骤4: 编译vLLM版本CUDA内核"
echo "=========================================="
if ! python compile_vllm.py; then
    echo "❌ vLLM版本CUDA内核编译失败"
    exit 1
fi

# 步骤5: 性能对比测试
echo "⚡ 步骤5: 性能对比测试"
echo "=========================================="
if ! python test_vllm_comparison.py; then
    echo "❌ 性能对比测试失败"
    exit 1
fi

echo "=========================================="
echo "🎉 CUDA内核测试完成!"
echo "🎯 目标性能: 0.10ms"
echo "=========================================="
