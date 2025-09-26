# 融合LN+QKV GPTQ CUDA内核

## 概述

本目录包含了融合LayerNorm + QKV投影的GPTQ量化CUDA内核实现，基于vLLM的优秀设计，实现了高效的融合算子。

## 文件结构

```
cuda/
├── ln_qkv_fusion_kernel.cu          # 融合LN+QKV CUDA内核源码
├── compile_ln_qkv_fusion.py         # Python编译脚本
├── compile_ln_qkv_fusion.sh         # Shell编译脚本
├── test_ln_qkv_fusion.py           # Python测试脚本
├── test_ln_qkv_fusion_all.sh       # 一键测试脚本
└── README_ln_qkv_fusion.md         # 本文件
```

## 核心特性

### 1. 融合算子
- **LayerNorm + QKV投影**: 将LayerNorm和QKV投影融合为单个CUDA内核
- **GPTQ 4bit量化**: 支持GPTQ 4bit量化权重
- **vLLM优化**: 基于vLLM的优秀实现

### 2. 性能优化
- **向量化计算**: 使用 `half2` 和 `__hfma2` 指令
- **高效分块**: 128x8的分块策略
- **共享内存**: 减少全局内存访问
- **原子操作**: 使用 `atomicAdd` 避免竞争条件

### 3. 目标性能
- **融合LN+QKV**: ≤ 0.25ms
- **加速比**: ≥ 2.0x (vs 分离算子)
- **内存效率**: 显著提升

## 使用方法

### 编译内核

**使用Python脚本**:
```bash
python3 compile_ln_qkv_fusion.py
```

**使用Shell脚本**:
```bash
bash compile_ln_qkv_fusion.sh
```

### 运行测试

**Python测试**:
```bash
python3 test_ln_qkv_fusion.py
```

**一键测试**:
```bash
bash test_ln_qkv_fusion_all.sh
```

## 测试内容

### 1. 基础功能测试
- 验证融合算子的基本功能
- 检查输出形状和数据类型
- 验证CUDA内核是否正确加载

### 2. 性能测试
- 测量融合算子的执行时间
- 计算统计信息（平均、最小、最大时间）
- 性能目标验证

### 3. 批处理测试
- 测试不同批处理配置
- 验证批处理性能
- 检查内存使用

### 4. 正确性测试
- 验证输出正确性
- 检查数值稳定性
- 检查输出范围

## 技术实现

### 1. 内核设计
```cpp
// 融合LayerNorm + GPTQ QKV投影内核（基于vLLM实现）
template <int m_count>
__global__ void fused_ln_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size, seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight,     // GPTQ量化权重 [K//8, N]
    const uint32_t* __restrict__ qzeros,      // GPTQ量化零点 [num_groups, groupsize//8]
    const half* __restrict__ scales,          // GPTQ量化缩放 [num_groups, N]
    const half* __restrict__ ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* __restrict__ ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* __restrict__ q_output,              // Q输出 [batch_size, num_heads, seq_len, head_size]
    half* __restrict__ k_output,              // K输出 [batch_size, kv_num_heads, seq_len, head_size]
    half* __restrict__ v_output,              // V输出 [batch_size, kv_num_heads, seq_len, head_size]
    // ... 其他参数
);
```

### 2. 优化技术
- **vLLM风格的分块策略**: `BLOCK_KN_SIZE=128`, `BLOCK_M_SIZE_MAX=8`
- **向量化点积**: 使用 `half2` 和 `__hfma2` 指令
- **高效4bit解量化**: 批量处理4bit权重
- **共享内存优化**: 缓存输入数据

### 3. 内存布局
- **输入**: `[batch_size, seq_len, hidden_dim]`
- **QKV输出**: `[batch_size, num_heads, seq_len, head_size]`
- **GPTQ权重**: `[hidden_dim//8, hidden_dim*3]`
- **LayerNorm参数**: `[hidden_dim]`

## 性能基准

### 目标性能
- **融合LN+QKV**: ≤ 0.25ms
- **GPTQ QKV**: ≤ 0.10ms
- **LayerNorm**: ≤ 0.05ms
- **加速比**: ≥ 2.0x (vs 分离算子)

### 测试配置
- **批处理大小**: [1, 2, 4, 8]
- **序列长度**: [1, 16, 32, 64]
- **隐藏维度**: [4096]
- **头数**: [32]
- **头大小**: [128]

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
```

## 故障排除

### 常见问题

1. **编译失败**
   ```
   ❌ 融合LN+QKV CUDA内核编译失败
   ```
   - 检查CUDA Toolkit安装
   - 检查编译环境
   - 检查内核代码

2. **性能不达标**
   ```
   ⚠️ 未达到性能目标: 0.30ms > 0.25ms
   ```
   - 检查GPU性能
   - 检查内核优化
   - 检查测试环境

3. **输出形状错误**
   ```
   ❌ Q形状错误: torch.Size([1, 32, 1, 128]) != torch.Size([1, 32, 1, 128])
   ```
   - 检查输入数据
   - 检查内核实现
   - 检查参数配置

### 调试技巧

1. **启用详细日志**
   ```bash
   python3 test_ln_qkv_fusion.py --verbose
   ```

2. **检查CUDA内核**
   ```bash
   ls -la fused_ln_qkv_gptq_cuda.so
   ```

3. **检查GPU状态**
   ```bash
   nvidia-smi
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

### 添加新功能
1. 修改 `ln_qkv_fusion_kernel.cu`
2. 更新测试脚本
3. 更新文档
4. 运行测试验证

### 报告问题
1. 提供测试日志
2. 提供环境信息
3. 提供复现步骤
4. 提供期望结果

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请联系项目维护者。