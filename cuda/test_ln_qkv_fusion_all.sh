#!/bin/bash
"""
融合LN+QKV GPTQ CUDA内核一键测试脚本
"""

echo "🚀 融合LN+QKV GPTQ CUDA内核一键测试开始"
echo "=========================================="

# 检查环境
echo "🔍 检查环境..."

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

# 检查PyTorch CUDA支持
echo "🔧 检查PyTorch CUDA支持..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch CUDA支持检查失败"
    exit 1
fi

# 显示GPU信息
echo "🔧 GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# 检查源文件
FUSION_FILE="ln_qkv_fusion_kernel.cu"
if [ ! -f "$FUSION_FILE" ]; then
    echo "❌ 源文件不存在: $FUSION_FILE"
    exit 1
fi

echo "✅ 源文件存在: $FUSION_FILE"

# 步骤1: 编译内核
echo "🔨 步骤1: 编译融合LN+QKV CUDA内核"
echo "=========================================="
python3 compile_ln_qkv_fusion.py

if [ $? -eq 0 ]; then
    echo "✅ 融合LN+QKV CUDA内核编译成功"
else
    echo "❌ 融合LN+QKV CUDA内核编译失败"
    exit 1
fi

# 步骤2: 运行测试
echo "🧪 步骤2: 运行融合LN+QKV CUDA内核测试"
echo "=========================================="
python3 test_ln_qkv_fusion.py

if [ $? -eq 0 ]; then
    echo "✅ 融合LN+QKV CUDA内核测试通过"
else
    echo "❌ 融合LN+QKV CUDA内核测试失败"
    exit 1
fi

# 步骤3: 性能对比测试
echo "⚡ 步骤3: 性能对比测试"
echo "=========================================="
echo "📊 运行性能对比测试..."

# 检查是否有分离算子测试
if [ -f "../test/test_performance_optimized.py" ]; then
    echo "📊 运行分离算子性能测试..."
    cd ../test
    python3 test_performance_optimized.py
    cd ../cuda
else
    echo "⚠️ 找不到分离算子性能测试，跳过对比"
fi

echo "=========================================="
echo "🎉 融合LN+QKV GPTQ CUDA内核测试完成!"
echo "📊 目标性能: ≤ 0.25ms"
echo "📊 目标加速比: ≥ 2.0x"
echo "=========================================="
