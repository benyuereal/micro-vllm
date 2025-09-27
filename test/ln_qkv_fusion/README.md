# LN+QKV融合内核测试

这个目录包含了融合LayerNorm + GPTQ QKV投影内核的测试套件。

## 文件结构

```
ln_qkv_fusion/
├── fusion.py                       # 主要的LN+QKV融合内核测试脚本（完整测试）
├── quick.py                        # 快速测试脚本（编译+功能+性能）
├── layer_integration.py            # 层集成测试
├── performance.py                  # 性能优化测试
└── README.md                       # 本文件
```

## 功能特性

- ✅ **完整的GPTQ INT4动态反量化**：包含qzeros处理
- ✅ **最优的LayerNorm实现**：使用Welford算法，数值稳定
- ✅ **QKV融合**：在同一个CUDA内核中执行LayerNorm和QKV投影
- ✅ **只使用float16**：不需要bfloat16转换
- ✅ **参考vLLM实现**：充分借鉴vLLM的优化策略

## 快速开始

### 运行完整测试

```bash
cd test/ln_qkv_fusion
python fusion.py
```

### 运行快速测试

```bash
python quick.py
```

### 运行层集成测试

```bash
python layer_integration.py
```

### 运行性能优化测试

```bash
python performance.py
```

## 测试内容

### 1. 功能测试 (`fusion.py`)

- **编译测试**：验证内核能够正常编译
- **功能测试**：验证基本功能正确性
- **精度测试**：与PyTorch实现对比精度
- **性能测试**：测量执行时间，目标 < 0.25ms

### 2. 快速测试 (`quick.py`)

- **编译+功能+性能**：一站式测试
- **快速验证**：适合开发调试
- **性能基准**：提供性能数据

### 3. 层集成测试 (`layer_integration.py`)

- **融合内核集成**：测试LayerNorm + GPTQ + QKV融合内核
- **端到端验证**：验证完整推理流程
- **形状检查**：验证QKV输出形状正确性

### 4. 性能优化测试 (`performance.py`)

- **融合内核性能**：测试LayerNorm + GPTQ + QKV融合性能
- **目标验证**：验证是否达到0.25ms目标
- **详细统计**：提供完整的性能分析

## 测试数据

### 默认配置
- **batch_size**: 1
- **seq_len**: 1
- **hidden_dim**: 4096
- **num_heads**: 32
- **kv_num_heads**: 32
- **head_size**: 128
- **groupsize**: 128

### 数据类型
- **输入**: `torch.float16`
- **权重**: `torch.uint32` (GPTQ量化)
- **缩放**: `torch.float16`
- **输出**: `torch.float16`

## 性能目标

- **平均执行时间**: < 0.25ms
- **内存使用**: 最小化
- **数值精度**: 与PyTorch实现一致
- **融合功能**: LayerNorm + GPTQ + QKV一体化

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

1. **首次运行**：建议先运行 `fusion.py` 进行完整测试
2. **开发调试**：使用 `quick.py` 进行快速验证
3. **集成测试**：使用 `layer_integration.py` 验证层集成
4. **性能分析**：使用 `performance.py` 进行详细分析