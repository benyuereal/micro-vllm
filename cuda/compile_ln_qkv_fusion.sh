#!/bin/bash
"""
融合LN+QKV GPTQ CUDA内核编译脚本
"""

echo "🚀 融合LN+QKV GPTQ CUDA内核编译开始"
echo "=========================================="

# 检查CUDA环境
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc未找到，请安装CUDA Toolkit"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 显示环境信息
echo "🔧 环境信息:"
echo "  CUDA版本: $(nvcc --version | grep release)"
echo "  Python版本: $(python3 --version)"
echo "  PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)')"

# 检查源文件
FUSION_FILE="ln_qkv_fusion_kernel.cu"
if [ ! -f "$FUSION_FILE" ]; then
    echo "❌ 源文件不存在: $FUSION_FILE"
    exit 1
fi

echo "✅ 源文件存在: $FUSION_FILE"

# 编译选项
NVCC_FLAGS="-O3 -use_fast_math -Xptxas=-O3 --ptxas-options=-v -maxrregcount=255 -lineinfo -std=c++17"

# 支持的GPU架构
ARCHS="compute_70,code=sm_70;compute_75,code=sm_75;compute_80,code=sm_80;compute_86,code=sm_86"

echo "🔨 开始编译融合LN+QKV CUDA内核..."

# 编译融合LN+QKV内核
python3 compile_ln_qkv_fusion.py

if [ $? -eq 0 ]; then
    echo "✅ 融合LN+QKV CUDA内核编译成功"
else
    echo "❌ 融合LN+QKV CUDA内核编译失败"
    exit 1
fi

echo "=========================================="
echo "🎉 融合LN+QKV GPTQ CUDA内核编译完成!"
echo "📊 目标性能: ≤ 0.25ms"
echo "📊 目标加速比: ≥ 2.0x"
echo "=========================================="