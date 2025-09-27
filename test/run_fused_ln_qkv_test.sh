#!/bin/bash
"""
融合LN+QKV GPTQ测试运行脚本
"""

echo "🚀 开始融合LN+QKV GPTQ测试"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查CUDA环境
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA驱动未安装"
    exit 1
fi

# 显示GPU信息
echo "🔧 GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# 检查PyTorch CUDA支持
echo "🔧 检查PyTorch CUDA支持..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

# 运行快速测试
echo "🧪 运行快速测试..."
python3 quick_fused_ln_qkv_test.py
if [ $? -eq 0 ]; then
    echo "✅ 快速测试通过"
else
    echo "❌ 快速测试失败"
    exit 1
fi

# 运行完整测试
echo "🧪 运行完整测试..."
python3 run_all_fused_ln_qkv_test.py
if [ $? -eq 0 ]; then
    echo "✅ 完整测试通过"
else
    echo "❌ 完整测试失败"
    exit 1
fi

echo "=========================================="
echo "🎉 所有测试完成!"
echo "📊 查看测试报告: fused_ln_qkv_test_report.md"
echo "📊 查看测试日志: fused_ln_qkv_test.log"
echo "=========================================="
