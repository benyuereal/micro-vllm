# vLLM融合内核测试

这个目录包含了vLLM GPTQ融合内核的测试套件。

## 文件结构

```
vllm_fusion/
├── fusion.py                  # 主要的vLLM融合内核功能测试脚本
├── quick.py                   # 快速测试脚本
├── performance.py             # 性能测试脚本
├── integration.py             # 集成测试脚本
└── README.md                  # 本文件
```

## 功能特性

- ✅ **vLLM GPTQ内核测试**：测试vLLM的GPTQ解量化功能
- ✅ **性能基准测试**：测量vLLM内核的执行时间
- ✅ **功能验证**：验证GPTQ解量化的正确性

## 快速开始

### 运行功能测试（推荐首次运行）

```bash
cd test/vllm_fusion
python fusion.py
```

### 运行快速测试

```bash
python quick.py
```

### 运行性能测试

```bash
python performance.py
```

### 运行集成测试

```bash
python integration.py
```

## 测试内容

### 1. 功能测试 (`fusion.py`)
- 验证vLLM GPTQ内核能够正常编译和运行
- 检查输入输出张量的形状和数据类型
- 验证GPTQ解量化的基本功能
- 包含完整的功能测试、精度测试和性能测试

### 2. 快速测试 (`quick.py`)
- 使用小矩阵尺寸进行快速验证
- 适合开发调试
- 提供快速的功能验证

### 3. 性能测试 (`performance.py`)
- 专注于性能分析和基准测试
- 测量vLLM内核的执行时间
- 提供详细的性能统计

### 4. 集成测试 (`integration.py`)
- 测试vLLM内核与现有系统的集成
- 端到端验证
- 测试GPTQCUDAFusion集成

## 依赖

- PyTorch >= 1.12
- CUDA >= 11.0
- Python >= 3.7

## 注意事项

- 确保CUDA环境正确配置
- 测试使用float16数据类型
- 支持动态groupsize测试
