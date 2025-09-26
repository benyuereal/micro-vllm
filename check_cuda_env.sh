#!/bin/bash
# 检查CUDA环境脚本

echo "🔍 检查CUDA环境..."

# 检查NVIDIA驱动
echo "📋 NVIDIA驱动:"
nvidia-smi

# 检查CUDA版本
echo "📋 CUDA版本:"
nvcc --version

# 检查PyTorch CUDA支持
echo "📋 PyTorch CUDA支持:"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 检查Triton
echo "📋 Triton版本:"
python -c "import triton; print(f'Triton版本: {triton.__version__}')"

# 检查CUDA文件
echo "📋 CUDA文件:"
ls -la core/layer/gptq_cuda_kernel.cu

echo "✅ CUDA环境检查完成!"
