# 融合LN+QKV GPTQ测试指南

## 概述

本目录包含了融合LayerNorm + QKV投影的GPTQ量化测试脚本，用于验证CUDA内核的功能、性能和正确性。

## 文件结构

```
├── run_all_fused_ln_qkv_test.py      # 完整测试脚本
├── quick_fused_ln_qkv_test.py       # 快速测试脚本
├── run_fused_ln_qkv_test.py         # 测试运行器
├── test_config_fused_ln_qkv.py      # 测试配置
└── README_fused_ln_qkv_test.md      # 本文件
```

## 测试类型

### 1. 基础功能测试
- 验证融合算子的基本功能
- 检查输出形状和数据类型
- 验证CUDA内核是否正确加载

### 2. 性能测试
- 测量融合算子的执行时间
- 与分离算子进行性能对比
- 计算加速比

### 3. 批处理测试
- 测试不同批处理配置
- 验证批处理性能
- 检查内存使用

### 4. 正确性测试
- 验证输出正确性
- 检查数值稳定性
- 与参考实现对比

### 5. 集成测试
- 测试与现有系统的集成
- 验证接口兼容性
- 检查错误处理

## 使用方法

### 快速测试

```bash
# 运行快速测试
python quick_fused_ln_qkv_test.py

# 或使用测试运行器
python run_fused_ln_qkv_test.py --mode quick
```

### 完整测试

```bash
# 运行所有测试
python run_all_fused_ln_qkv_test.py

# 或使用测试运行器
python run_fused_ln_qkv_test.py --mode full
```

### 特定测试

```bash
# 运行特定测试
python run_fused_ln_qkv_test.py --mode specific --test basic
python run_fused_ln_qkv_test.py --mode specific --test performance
python run_fused_ln_qkv_test.py --mode specific --test batch
python run_fused_ln_qkv_test.py --mode specific --test correctness
python run_fused_ln_qkv_test.py --mode specific --test integration
```

### 详细输出

```bash
# 启用详细输出
python run_fused_ln_qkv_test.py --mode full --verbose
```

## 测试配置

测试配置在 `test_config_fused_ln_qkv.py` 中定义：

### 性能目标
- 融合LN+QKV: ≤ 0.25ms
- GPTQ QKV: ≤ 0.10ms
- LayerNorm: ≤ 0.05ms
- 加速比: ≥ 2.0x (vs 分离算子)

### 测试数据配置
- 批处理大小: [1, 2, 4, 8]
- 序列长度: [1, 16, 32, 64]
- 隐藏维度: [4096]

### 性能测试配置
- 运行次数: 100
- 预热次数: 10
- 批处理测试次数: 50

## 环境要求

### 硬件要求
- NVIDIA GPU (计算能力 ≥ 7.0)
- GPU内存 ≥ 4GB
- CUDA版本 ≥ 11.0

### 软件要求
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- CUDA Toolkit ≥ 11.0

### 依赖库
```bash
pip install torch torchvision torchaudio
pip install numpy
pip install logging
```

## 测试结果

### 输出文件
- `fused_ln_qkv_test.log`: 测试日志
- `fused_ln_qkv_test_report.md`: 测试报告

### 测试报告示例
```markdown
# 融合LN+QKV GPTQ测试报告

## 测试环境
- CUDA设备: NVIDIA GeForce RTX 4090
- CUDA版本: 12.1
- PyTorch版本: 2.0.1

## 测试结果
- 基础功能: ✅ 通过
- 性能测试: ✅ 通过
- 批处理测试: ✅ 通过
- 正确性测试: ✅ 通过
- 集成测试: ✅ 通过

## 总结
- 通过率: 5/5 (100.0%)
- 状态: 🎉 所有测试通过
```

## 故障排除

### 常见问题

1. **CUDA不可用**
   ```
   ❌ CUDA不可用，无法运行测试
   ```
   - 检查CUDA安装
   - 检查GPU驱动
   - 检查PyTorch CUDA支持

2. **内核编译失败**
   ```
   ❌ CUDA内核编译失败
   ```
   - 检查CUDA Toolkit安装
   - 检查编译环境
   - 检查内核代码

3. **性能不达标**
   ```
   ⚠️ 未达到性能目标: 0.30ms > 0.25ms
   ```
   - 检查GPU性能
   - 检查内核优化
   - 检查测试环境

4. **输出形状错误**
   ```
   ❌ Q形状错误: torch.Size([1, 32, 1, 128]) != torch.Size([1, 32, 1, 128])
   ```
   - 检查输入数据
   - 检查内核实现
   - 检查参数配置

### 调试技巧

1. **启用详细日志**
   ```bash
   python run_fused_ln_qkv_test.py --mode full --verbose
   ```

2. **检查CUDA内核**
   ```bash
   ls -la cuda/ln_qkv_fusion_kernel.so
   ```

3. **检查GPU状态**
   ```bash
   nvidia-smi
   ```

4. **检查PyTorch CUDA**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

## 性能优化

### 内核优化
- 使用共享内存
- 向量化操作
- 减少内存访问
- 优化线程配置

### 测试优化
- 预热GPU
- 减少测试开销
- 优化内存使用
- 并行测试

## 贡献指南

### 添加新测试
1. 在 `test_config_fused_ln_qkv.py` 中添加配置
2. 在 `run_all_fused_ln_qkv_test.py` 中添加测试方法
3. 更新测试运行器
4. 更新文档

### 报告问题
1. 提供测试日志
2. 提供环境信息
3. 提供复现步骤
4. 提供期望结果

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请联系项目维护者。
