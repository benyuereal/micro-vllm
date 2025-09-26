# CUDA内核目录

本目录包含CUDA C++内核的实现、编译和测试。

## 目录结构

```
cuda/
├── src/                    # CUDA源码
│   ├── gptq_cuda_kernel.cu # CUDA内核源码
│   └── gptq_cuda.py        # CUDA内核Python接口
├── tests/                  # 测试文件
│   └── test_cuda_simple.py # CUDA内核测试
├── scripts/                # 脚本文件
│   ├── compile_cuda_kernel.py # 编译脚本
│   ├── test_cuda_all.sh    # 一键测试脚本
│   └── check_cuda_env.sh   # 环境检查脚本
├── setup_cuda.py           # CUDA编译配置
└── README.md               # 本文件
```

## 使用方法

### 1. 检查CUDA环境
```bash
cd cuda
bash scripts/check_cuda_env.sh
```

### 2. 编译CUDA内核
```bash
python scripts/compile_cuda_kernel.py
```

### 3. 测试CUDA内核
```bash
python tests/test_cuda_simple.py
```

### 4. 一键测试
```bash
bash scripts/test_cuda_all.sh
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
