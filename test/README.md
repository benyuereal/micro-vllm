# 测试目录

本目录包含GPTQ融合算子的各种测试脚本。

## 📁 测试文件说明

### 🚀 核心功能测试
- **`test_qwen7b_gptq_fast.py`** - 快速测试Qwen7B GPTQ格式修复
  - 使用小矩阵快速验证逻辑
  - 包含QKV投影和注意力结构测试
  - 运行时间短，适合快速验证

- **`test_qwen7b_structure.py`** - 完整Qwen7B模型结构测试
  - 测试完整的Qwen7B模型结构
  - 验证QKV投影和注意力机制
  - 确保所有维度正确

### 🔧 专门功能测试
- **`test_output_projection_gptq_fix.py`** - 输出投影GPTQ格式修复测试
  - 专门测试输出投影的GPTQ格式
  - 验证两种不同格式的处理
  - 确保输出投影正确工作

- **`test_output_projection_shape.py`** - 输出投影形状处理测试
  - 测试注意力输出形状重塑
  - 验证数据一致性
  - 确保形状转换正确

### ⚡ 性能测试
- **`test_large_matrix.py`** - 大矩阵性能测试
  - 测试GPTQ性能与大型矩阵
  - 验证内存使用和计算效率
  - 适合性能基准测试

- **`test_optimization.py`** - 优化方法测试
  - 比较不同反量化优化方法
  - 验证性能提升效果
  - 确保优化正确性

## 🎯 推荐测试顺序

1. **快速功能验证**:
   ```bash
   cd test && python test_qwen7b_gptq_fast.py
   ```

2. **输出投影测试**:
   ```bash
   cd test && python test_output_projection_gptq_fix.py
   ```

3. **完整结构测试**:
   ```bash
   cd test && python test_qwen7b_structure.py
   ```

4. **性能测试** (可选):
   ```bash
   cd test && python test_large_matrix.py
   cd test && python test_optimization.py
   ```

## 📊 测试结果

所有测试通过后，你应该看到：
- ✅ QKV投影输出维度: 12288
- ✅ 输出投影输出维度: 4096
- ✅ 所有形状转换正确
- ✅ GPTQ格式处理正确

## 🔧 故障排除

如果测试失败，请检查：
1. CUDA是否可用
2. GPTQ参数格式是否正确
3. 输入输出维度是否匹配
4. 日志中的错误信息