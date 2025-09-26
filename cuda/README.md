# CUDA内核目录

本目录包含CUDA C++内核的实现、编译和测试。

## 目录结构

```
cuda/
├── gptq_cuda_kernel.cu        # CUDA内核源码（向量化优化）
├── gptq_cuda.py               # CUDA内核Python接口
├── compile.py                 # 编译脚本
├── test.py                    # 测试脚本
└── README.md                  # 本文件
```

## 使用方法

### 1. 编译CUDA内核
```bash
python compile.py
```

### 2. 测试CUDA内核
```bash
python test.py
```

## 环境要求

- NVIDIA GPU
- CUDA Toolkit 12.0+
- PyTorch with CUDA support
- Python 3.8+

## 性能目标

- QKV投影: < 0.1ms
- 输出投影: < 0.2ms
- 总体加速: > 10x vs PyTorch

## 优化特性

- ✅ **向量化运算**: 使用half2向量化处理
- ✅ **内存优化**: 减少内存访问次数
- ✅ **寄存器优化**: 高效使用GPU寄存器
- ✅ **简化结构**: 扁平化目录结构
