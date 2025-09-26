# GPTQ测试文件说明

## 📁 测试文件结构

### 🎯 核心测试文件

#### `test_comprehensive.py` - 综合测试（推荐）
**用途**: 包含所有重要测试功能的综合测试文件
**包含测试**:
- ✅ 混合GPTQ格式测试（实际推理场景）
- ✅ 标准GPTQ格式测试
- ✅ 性能优化效果测试
- ✅ 正确性验证测试

**运行方式**:
```bash
cd test && python test_comprehensive.py
```

#### `test_optimization.py` - 优化效果测试
**用途**: 专门测试不同反量化方法的性能对比
**特点**: 详细的性能分析和加速比计算

**运行方式**:
```bash
cd test && python test_optimization.py
```

#### `test_large_matrix.py` - 大矩阵性能测试
**用途**: 测试大矩阵尺寸下的GPTQ性能
**特点**: 使用大矩阵（1024x4096x2048）进行压力测试

**运行方式**:
```bash
cd test && python test_large_matrix.py
```

## 🚀 快速开始

### 1. 运行综合测试（推荐）
```bash
cd test && python test_comprehensive.py
```

### 2. 运行性能测试
```bash
cd test && python test_optimization.py
```

### 3. 运行大矩阵测试
```bash
cd test && python test_large_matrix.py
```

## 📊 测试说明

### 测试环境要求
- CUDA可用
- PyTorch已安装
- Triton已安装（可选，用于融合算子）

### 测试内容
1. **格式兼容性**: 测试各种GPTQ格式的兼容性
2. **性能优化**: 比较不同反量化方法的性能
3. **正确性验证**: 确保融合算子与基线实现结果一致
4. **大矩阵支持**: 验证大矩阵尺寸下的稳定性

### 预期结果
- ✅ 所有格式测试通过
- ✅ 性能提升明显（2-5倍加速）
- ✅ 正确性误差 < 1e-3
- ✅ 大矩阵测试稳定运行

## 🔧 故障排除

### 常见问题
1. **CUDA不可用**: 检查CUDA环境配置
2. **Triton导入错误**: 检查Triton安装
3. **内存不足**: 减少矩阵尺寸或使用CPU测试

### 调试建议
- 查看详细日志输出
- 检查矩阵维度匹配
- 验证GPTQ格式检测逻辑
