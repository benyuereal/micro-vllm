# Test目录说明

## 目录结构
```
test/
├── test_gptq_functionality.py    # GPTQ功能测试
├── test_batch_layer_shapes.py    # 批处理Layer层测试
├── test_layer_integration.py     # Layer层集成测试
├── run_all_tests.py              # 一键测试脚本
└── README.md                     # 测试说明
```

## 测试说明

### 1. GPTQ功能测试 (`test_gptq_functionality.py`)
- 测试CUDA融合内核的基本功能
- 验证QKV投影、注意力输出投影、输出投影、MLP投影
- 性能基准测试
- 目标：0.10ms延迟

### 2. 批处理Layer层测试 (`test_batch_layer_shapes.py`)
- 专门测试批处理layer层形状
- 验证3D到2D形状转换
- 完整的layer层数据流测试
- 基于实际layer层数据形状

### 3. Layer层集成测试 (`test_layer_integration.py`)
- 测试optimized_qwen_layer.py与CUDA融合内核的集成
- 验证实际layer层数据形状
- 测试前向传播

### 4. 一键测试脚本 (`run_all_tests.py`)
- 运行所有测试
- 测试结果汇总

## 使用方法

### 运行单个测试
```bash
cd test
python test_gptq_functionality.py
python test_batch_layer_shapes.py
python test_layer_integration.py
```

### 运行所有测试
```bash
cd test
python run_all_tests.py
```

## 测试数据形状

基于实际layer层数据形状：
- **QKV投影**: input[1,4096] -> output[1,12288]
- **注意力输出投影**: input[32,128] -> output[32,128]
- **输出投影**: input[1,4096] -> output[1,4096]
- **MLP投影1**: input[1,4096] -> output[1,11008]
- **MLP投影2**: input[1,11008] -> output[1,4096]

## 性能目标

- **目标延迟**: 0.20ms
- **当前性能**: 0.14-0.19ms (已接近目标)
- **优化策略**: 无回退，只有向前优化

## 注意事项

1. 需要CUDA环境
2. 需要编译CUDA内核
3. 测试前确保gptq.py已更新
4. 如果测试失败，检查CUDA内核编译状态