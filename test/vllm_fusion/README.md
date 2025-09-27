# vLLM融合内核测试

这个目录包含了vLLM GPTQ融合内核的测试套件。

## 文件结构

```
vllm_fusion/
├── vllm_fusion.py             # 主要的vLLM融合内核测试脚本
├── quick_gptq.py              # 快速GPTQ测试
├── gptq_functionality.py      # GPTQ功能测试
└── README.md                  # 本文件
```

## 功能特性

- ✅ **vLLM GPTQ内核测试**：测试vLLM的GPTQ解量化功能
- ✅ **性能基准测试**：测量vLLM内核的执行时间
- ✅ **功能验证**：验证GPTQ解量化的正确性

## 快速开始

### 运行vLLM融合内核测试

```bash
cd test/vllm_fusion
python vllm_fusion.py
```

### 运行快速GPTQ测试

```bash
python quick_gptq.py
```

### 运行GPTQ功能测试

```bash
python gptq_functionality.py
```

## 测试内容

### 1. 功能测试
- 验证vLLM GPTQ内核能够正常编译和运行
- 检查输入输出张量的形状和数据类型
- 验证GPTQ解量化的基本功能

### 2. 性能测试
- 测量vLLM内核的执行时间
- 提供性能基准数据
- 支持不同规模的测试数据

## 依赖

- PyTorch >= 1.12
- CUDA >= 11.0
- Python >= 3.7

## 注意事项

- 确保CUDA环境正确配置
- 测试使用float16数据类型
- 支持动态groupsize测试
