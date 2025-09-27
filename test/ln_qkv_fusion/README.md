# LN+QKV融合内核测试

这个目录包含了融合LayerNorm + GPTQ QKV投影内核的测试套件。

## 文件结构

```
ln_qkv_fusion/
├── test_ln_qkv_fusion.py     # 主要的LN+QKV融合内核测试脚本（完整测试）
├── test_quick.py             # 快速测试脚本（编译+功能+性能）
├── test_layer_integration.py # 层集成测试
├── test_performance_optimized.py # 性能优化测试
└── README.md                 # 本文件
```

## 功能特性

- ✅ **完整的GPTQ INT4动态反量化**：包含qzeros处理
- ✅ **最优的LayerNorm实现**：使用Welford算法，数值稳定
- ✅ **QKV融合**：在同一个CUDA内核中执行LayerNorm和QKV投影
- ✅ **只使用float16**：不需要bfloat16转换
- ✅ **参考vLLM实现**：充分借鉴vLLM的优化策略

## 快速开始

### 运行快速测试（推荐）

```bash
cd test/ln_qkv_fusion
python test_quick.py
```

### 运行完整测试

```bash
python test_ln_qkv_fusion.py
```

### 运行层集成测试

```bash
python test_layer_integration.py
```

### 运行性能优化测试

```bash
python test_performance_optimized.py
```

## 测试内容

### 1. 快速测试 (`test_quick.py`)
- **编译测试**：验证内核能够正常编译
- **功能测试**：验证基本功能正确性
- **性能测试**：测量执行时间（目标：< 0.5ms）

### 2. 完整测试 (`test_ln_qkv_fusion.py`)
- **基础功能测试**：验证融合内核能够正常编译和运行
- **LayerNorm精度测试**：与PyTorch LayerNorm对比精度
- **GPTQ解量化测试**：验证GPTQ INT4解量化功能
- **性能测试**：测量融合内核执行时间

### 3. 层集成测试 (`test_layer_integration.py`)
- 测试与现有层的集成
- 验证端到端功能
- 检查内存使用情况

### 4. 性能优化测试 (`test_performance_optimized.py`)
- 详细的性能分析
- 不同规模数据的性能测试
- 优化建议

## 内核参数

```cpp
void fused_ln_qkv_gptq_cuda(
    const half* input,           // [batch_size, seq_len, hidden_dim]
    const uint32_t* qweight,     // GPTQ量化权重 [K//8, N]
    const uint32_t* qzeros,      // GPTQ量化零点 [num_groups, groupsize//8]
    const half* scales,          // GPTQ量化缩放 [num_groups, N]
    const half* ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* q_output,              // Q输出 [batch_size, num_heads, seq_len, head_size]
    half* k_output,              // K输出 [batch_size, kv_num_heads, seq_len, head_size]
    half* v_output,              // V输出 [batch_size, kv_num_heads, seq_len, head_size]
    int batch_size, int seq_len, int hidden_dim,
    int num_heads, int kv_num_heads, int head_size,
    int groupsize, float eps
);
```

## 性能优化

- 使用vLLM的分块策略：`BLOCK_KN_SIZE=128`, `BLOCK_M_SIZE_MAX=8`
- 使用vLLM的向量化技术：`half2` 和 `__hfma2`
- 使用共享内存优化
- 使用原子操作避免竞争条件

## 依赖

- PyTorch >= 1.12
- CUDA >= 11.0
- Python >= 3.7

## 注意事项

- 确保CUDA环境正确配置
- 内核使用float16数据类型
- 支持动态groupsize
- 包含完整的GPTQ解量化逻辑

## 使用建议

1. **首次使用**：建议先运行 `test_quick.py` 进行快速验证
2. **开发调试**：使用 `test_ln_qkv_fusion.py` 进行完整测试
3. **性能分析**：使用 `test_performance_optimized.py` 进行详细分析
4. **集成测试**：使用 `test_layer_integration.py` 验证集成功能