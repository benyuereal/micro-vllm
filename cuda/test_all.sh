#!/bin/bash
"""
CUDA内核一键测试 - 包括GPTQ和融合LN+QKV内核
"""

echo "🚀 CUDA内核一键测试开始..."
echo "=========================================="
echo "🎯 目标性能: 0.10ms (GPTQ), 0.25ms (融合LN+QKV)"
echo "=========================================="

# 步骤1: 检查环境
echo "🔍 步骤1: 检查CUDA环境"
echo "=========================================="
if ! python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" 2>/dev/null; then
    echo "❌ CUDA环境检查失败"
    exit 1
fi

# 步骤2: 编译GPTQ内核
echo "🔨 步骤2: 编译GPTQ内核"
echo "=========================================="
if [ -f "compile_gptq_kernel.sh" ]; then
    bash compile_gptq_kernel.sh
    if [ $? -eq 0 ]; then
        echo "✅ GPTQ内核编译成功"
    else
        echo "❌ GPTQ内核编译失败"
        exit 1
    fi
else
    echo "⚠️ 找不到GPTQ内核编译脚本"
fi

# 步骤3: 编译融合LN+QKV内核
echo "🔨 步骤3: 编译融合LN+QKV内核"
echo "=========================================="
if [ -f "compile_ln_qkv_fusion.sh" ]; then
    bash compile_ln_qkv_fusion.sh
    if [ $? -eq 0 ]; then
        echo "✅ 融合LN+QKV内核编译成功"
    else
        echo "❌ 融合LN+QKV内核编译失败"
        exit 1
    fi
else
    echo "⚠️ 找不到融合LN+QKV内核编译脚本"
fi

# 步骤4: 运行GPTQ功能测试
echo "🧪 步骤4: 运行GPTQ功能测试"
echo "=========================================="
if [ -f "../test/test_gptq_functionality.py" ]; then
    cd ../test
    python test_gptq_functionality.py
    if [ $? -eq 0 ]; then
        echo "✅ GPTQ功能测试通过"
    else
        echo "❌ GPTQ功能测试失败"
        exit 1
    fi
    cd ../cuda
else
    echo "⚠️ 找不到GPTQ功能测试脚本"
fi

# 步骤5: 运行融合LN+QKV功能测试
echo "🧪 步骤5: 运行融合LN+QKV功能测试"
echo "=========================================="
if [ -f "../test/test_fused_ln_qkv_gptq.py" ]; then
    cd ../test
    python test_fused_ln_qkv_gptq.py
    if [ $? -eq 0 ]; then
        echo "✅ 融合LN+QKV功能测试通过"
    else
        echo "❌ 融合LN+QKV功能测试失败"
        exit 1
    fi
    cd ../cuda
else
    echo "⚠️ 找不到融合LN+QKV功能测试脚本"
fi

# 步骤6: 运行性能测试
echo "🚀 步骤6: 运行性能测试"
echo "=========================================="
if [ -f "../test/test_fused_ln_qkv_gptq_performance.py" ]; then
    cd ../test
    python test_fused_ln_qkv_gptq_performance.py
    if [ $? -eq 0 ]; then
        echo "✅ 性能测试通过"
    else
        echo "❌ 性能测试失败"
        exit 1
    fi
    cd ../cuda
else
    echo "⚠️ 找不到性能测试脚本"
fi

# 步骤7: 运行集成测试
echo "🔗 步骤7: 运行集成测试"
echo "=========================================="
if [ -f "../test/test_fused_ln_qkv_gptq_integration.py" ]; then
    cd ../test
    python test_fused_ln_qkv_gptq_integration.py
    if [ $? -eq 0 ]; then
        echo "✅ 集成测试通过"
    else
        echo "❌ 集成测试失败"
        exit 1
    fi
    cd ../cuda
else
    echo "⚠️ 找不到集成测试脚本"
fi

echo "=========================================="
echo "🎉 所有CUDA内核测试完成!"
echo "🎯 GPTQ目标: 0.10ms, 融合LN+QKV目标: 0.25ms"
echo "=========================================="
