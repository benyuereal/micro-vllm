# 测试目录

这个目录包含了所有融合内核的测试套件，按功能分类组织。

## 目录结构

```
test/
├── vllm_fusion/              # vLLM融合内核测试
│   ├── test_vllm_fusion.py   # 主要的vLLM融合内核测试脚本
│   ├── quick_gptq_test.py    # 快速GPTQ测试
│   ├── test_gptq_functionality.py # GPTQ功能测试
│   └── README.md             # vLLM融合测试说明
├── ln_qkv_fusion/            # LN+QKV融合内核测试
│   ├── test_ln_qkv_fusion.py # 主要的LN+QKV融合内核测试脚本
│   ├── quick_fused_ln_qkv_test.py # 快速融合测试
│   ├── run_all_fused_ln_qkv_test.py # 运行所有融合测试
│   ├── run_fused_ln_qkv_test.py # 运行融合测试
│   ├── run_fused_ln_qkv_test.sh # 融合测试shell脚本
│   ├── test_config_fused_ln_qkv.py # 融合配置测试
│   ├── test_performance_optimized.py # 性能优化测试
│   ├── test_layer_integration.py # 层集成测试
│   └── README.md             # LN+QKV融合测试说明
├── run_all_tests.py          # 主测试脚本 - 运行所有测试
├── analyze_performance.py    # 性能分析脚本
├── quick_test.py             # 快速测试脚本
└── README.md                 # 本文件
```

## 快速开始

### 运行所有测试

```bash
cd test
python run_all_tests.py
```

### 运行vLLM融合内核测试

```bash
cd test/vllm_fusion
python test_vllm_fusion.py
```

### 运行LN+QKV融合内核测试

```bash
cd test/ln_qkv_fusion
python test_ln_qkv_fusion.py
```

## 测试分类

### vLLM融合内核测试 (`vllm_fusion/`)

- **功能**：测试vLLM的GPTQ解量化功能
- **特性**：
  - vLLM GPTQ内核测试
  - 性能基准测试
  - 功能验证
- **目标**：验证vLLM内核的正确性和性能

### LN+QKV融合内核测试 (`ln_qkv_fusion/`)

- **功能**：测试融合LayerNorm + GPTQ QKV投影内核
- **特性**：
  - 完整的GPTQ INT4动态反量化
  - 最优的LayerNorm实现
  - QKV融合
  - 只使用float16
  - 参考vLLM实现
- **目标**：验证融合内核的功能、精度和性能

## 测试内容

### 1. 基础功能测试
- 验证内核能够正常编译和运行
- 检查输入输出张量的形状和数据类型
- 验证基本功能

### 2. 精度测试
- 与PyTorch实现对比精度
- 验证数值稳定性
- 检查算法正确性

### 3. 性能测试
- 测量内核执行时间
- 提供性能基准数据
- 支持不同规模的测试数据

### 4. 集成测试
- 测试与现有层的集成
- 验证端到端功能
- 检查内存使用情况

## 依赖

- PyTorch >= 1.12
- CUDA >= 11.0
- Python >= 3.7

## 注意事项

- 确保CUDA环境正确配置
- 测试使用float16数据类型
- 支持动态groupsize测试
- 包含完整的GPTQ解量化逻辑

## 使用建议

1. **首次运行**：建议先运行 `run_all_tests.py` 来验证所有功能
2. **开发调试**：使用对应目录下的具体测试脚本
3. **性能分析**：使用 `analyze_performance.py` 进行详细分析
4. **快速验证**：使用 `quick_test.py` 进行快速功能验证