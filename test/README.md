# 测试目录说明

## 目录结构

```
test/
├── 核心功能测试
│   ├── test_gptq_functionality.py        # GPTQ功能测试
│   ├── test_batch_layer_shapes.py        # 批处理Layer层测试
│   ├── test_layer_integration.py         # Layer层集成测试
│   └── test_performance_optimized.py     # 性能优化测试
├── 快速测试
│   ├── quick_test.py                     # 快速API服务器测试
│   ├── quick_fused_ln_qkv_test.py        # 快速融合LN+QKV测试
│   └── quick_gptq_test.py                # 快速GPTQ测试
├── 融合LN+QKV测试
│   ├── run_fused_ln_qkv_test.py          # 融合LN+QKV测试
│   ├── run_fused_ln_qkv_test.sh          # 融合LN+QKV测试Shell脚本
│   └── run_all_fused_ln_qkv_test.py      # 运行所有融合LN+QKV测试
├── 配置测试
│   └── test_config_fused_ln_qkv.py       # 融合LN+QKV配置测试
├── 性能分析
│   └── analyze_performance.py            # 性能分析脚本
├── 一键测试
│   └── run_all_tests.py                  # 一键测试脚本
└── README.md                             # 本文件
```

## 测试分类

### 1. 核心功能测试

#### GPTQ功能测试 (`test_gptq_functionality.py`)
- 测试CUDA融合内核的基本功能
- 验证QKV投影、注意力输出投影、输出投影、MLP投影
- 性能基准测试
- 目标：0.10ms延迟

#### 批处理Layer层测试 (`test_batch_layer_shapes.py`)
- 专门测试批处理layer层形状
- 验证3D到2D形状转换
- 完整的layer层数据流测试
- 基于实际layer层数据形状

#### Layer层集成测试 (`test_layer_integration.py`)
- 测试optimized_qwen_layer.py与CUDA融合内核的集成
- 验证实际layer层数据形状
- 测试前向传播

#### 性能优化测试 (`test_performance_optimized.py`)
- 测试格式缓存和groupsize优化
- 验证不同groupsize的性能
- 找出最佳性能配置

### 2. 快速测试

#### 快速API服务器测试 (`quick_test.py`)
- 简单测试验证API服务器格式修复
- 快速验证基本功能

#### 快速融合LN+QKV测试 (`quick_fused_ln_qkv_test.py`)
- 快速测试融合LN+QKV功能
- 验证基本正确性

#### 快速GPTQ测试 (`quick_gptq_test.py`)
- 快速测试GPTQ功能
- 验证基本正确性

### 3. 融合LN+QKV测试

#### 融合LN+QKV测试 (`run_fused_ln_qkv_test.py`)
- 完整的融合LN+QKV功能测试
- 性能测试和正确性验证

#### 融合LN+QKV测试Shell脚本 (`run_fused_ln_qkv_test.sh`)
- Shell版本的融合LN+QKV测试
- 适合自动化脚本

#### 运行所有融合LN+QKV测试 (`run_all_fused_ln_qkv_test.py`)
- 运行所有融合LN+QKV相关测试
- 测试结果汇总

### 4. 配置测试

#### 融合LN+QKV配置测试 (`test_config_fused_ln_qkv.py`)
- 测试融合LN+QKV的配置参数
- 验证不同配置下的性能

### 5. 性能分析

#### 性能分析脚本 (`analyze_performance.py`)
- 分析测试结果
- 生成性能报告
- 识别性能瓶颈

### 6. 一键测试

#### 一键测试脚本 (`run_all_tests.py`)
- 运行所有核心测试
- 测试结果汇总
- 生成测试报告

## 使用方法

### 运行单个测试

**核心功能测试**:
```bash
cd test
python test_gptq_functionality.py
python test_batch_layer_shapes.py
python test_layer_integration.py
python test_performance_optimized.py
```

**快速测试**:
```bash
cd test
python quick_test.py
python quick_fused_ln_qkv_test.py
python quick_gptq_test.py
```

**融合LN+QKV测试**:
```bash
cd test
python run_fused_ln_qkv_test.py
bash run_fused_ln_qkv_test.sh
python run_all_fused_ln_qkv_test.py
```

**配置测试**:
```bash
cd test
python test_config_fused_ln_qkv.py
```

**性能分析**:
```bash
cd test
python analyze_performance.py
```

### 运行所有测试

**一键测试**:
```bash
cd test
python run_all_tests.py
```

**融合LN+QKV一键测试**:
```bash
cd test
python run_all_fused_ln_qkv_test.py
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

## 环境要求

1. **CUDA环境**: 需要CUDA Toolkit和GPU
2. **Python依赖**: PyTorch, NumPy等
3. **编译状态**: 需要先编译CUDA内核
4. **模型文件**: 需要GPTQ量化模型文件

## 注意事项

1. 测试前确保CUDA内核已编译
2. 确保gptq.py已更新到最新版本
3. 如果测试失败，检查CUDA内核编译状态
4. 性能测试需要GPU环境
5. 某些测试可能需要特定的模型文件

## 故障排除

### 常见问题

1. **CUDA内核未编译**
   ```
   ❌ 无法导入CUDA内核
   ```
   - 运行 `cd cuda && python compile_vllm.py`
   - 运行 `cd cuda && python compile_ln_qkv_fusion.py`

2. **性能不达标**
   ```
   ⚠️ 性能未达到目标
   ```
   - 检查GPU性能
   - 检查内核优化
   - 检查测试环境

3. **测试失败**
   ```
   ❌ 测试失败
   ```
   - 检查日志输出
   - 检查环境配置
   - 检查模型文件

### 调试技巧

1. **启用详细日志**
   ```bash
   python test_gptq_functionality.py --verbose
   ```

2. **检查CUDA状态**
   ```bash
   nvidia-smi
   ```

3. **检查编译状态**
   ```bash
   ls -la cuda/fused_*.so
   ```