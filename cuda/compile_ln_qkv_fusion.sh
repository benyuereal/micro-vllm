#!/bin/bash

# 融合LN+QKV GPTQ内核编译脚本
# 编译CUDA内核为共享库

set -e

# 配置
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC=${NVCC:-${CUDA_HOME}/bin/nvcc}
PYTHON_INCLUDE=${PYTHON_INCLUDE:-$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")}
TORCH_LIB=${TORCH_LIB:-$(python -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])")}

# 编译参数
CUDA_ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
NVCC_FLAGS="-O3 -use_fast_math -lineinfo --ptxas-options=-v"
INCLUDE_FLAGS="-I${CUDA_HOME}/include -I${PYTHON_INCLUDE}"
LIBRARY_FLAGS="-L${CUDA_HOME}/lib64 -L${TORCH_LIB}"

# 源文件
SOURCE_FILE="ln_qkv_fusion_kernel.cu"
OUTPUT_FILE="ln_qkv_fusion_kernel.so"

echo "🔧 编译融合LN+QKV GPTQ内核..."
echo "📁 源文件: ${SOURCE_FILE}"
echo "📁 输出文件: ${OUTPUT_FILE}"
echo "🔧 CUDA_HOME: ${CUDA_HOME}"
echo "🔧 NVCC: ${NVCC}"

# 检查CUDA环境
if [ ! -f "${NVCC}" ]; then
    echo "❌ 找不到NVCC: ${NVCC}"
    echo "请设置正确的CUDA_HOME环境变量"
    exit 1
fi

# 检查源文件
if [ ! -f "${SOURCE_FILE}" ]; then
    echo "❌ 找不到源文件: ${SOURCE_FILE}"
    exit 1
fi

# 编译
echo "🚀 开始编译..."
${NVCC} ${CUDA_ARCH} ${NVCC_FLAGS} ${INCLUDE_FLAGS} ${LIBRARY_FLAGS} \
    --shared -Xcompiler -fPIC \
    ${SOURCE_FILE} -o ${OUTPUT_FILE}

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "📁 输出文件: ${OUTPUT_FILE}"
    
    # 检查输出文件
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "📊 文件大小: $(ls -lh ${OUTPUT_FILE} | awk '{print $5}')"
        echo "🎯 编译完成!"
    else
        echo "❌ 编译失败: 找不到输出文件"
        exit 1
    fi
else
    echo "❌ 编译失败!"
    exit 1
fi

echo "🎉 融合LN+QKV GPTQ内核编译完成!"
