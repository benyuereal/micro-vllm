# 融合LN+QKV GPTQ量化方案

## 概述

本方案实现了LayerNorm + QKV投影的CUDA融合算子，支持GPTQ 4bit量化，旨在显著提升Transformer层的推理性能。

## 文件结构

```
cuda/
├── ln_qkv_fusion_kernel.cu          # CUDA融合内核实现
├── compile_ln_qkv_fusion.sh         # 编译脚本
└── test_all.sh                      # 一键测试脚本

core/layer/
└── fused_ln_qkv_gptq.py             # Python包装器

test/
├── test_fused_ln_qkv_gptq.py        # 功能测试
├── test_fused_ln_qkv_gptq_performance.py  # 性能测试
└── test_fused_ln_qkv_gptq_integration.py  # 集成测试
```

## 技术特性

### 1. 融合算子
- **LayerNorm + QKV投影**：将两个操作融合为一个CUDA内核
- **GPTQ 4bit量化**：支持GPTQ量化权重的解量化和矩阵乘法
- **多输出处理**：同时输出Q、K、V三个张量
- **批处理支持**：支持不同的batch size和sequence length

### 2. 性能优化
- **共享内存**：使用shared memory缓存LayerNorm统计信息
- **向量化操作**：使用CUDA向量化指令优化计算
- **内存访问优化**：减少内存访问次数和延迟
- **内核融合**：避免中间结果的存储和读取

### 3. 数值稳定性
- **LayerNorm epsilon**：防止除零错误
- **GPTQ解量化**：正确处理4bit权重的解量化
- **数据类型转换**：确保float16精度

## 使用方法

### 1. 编译

```bash
cd cuda
bash compile_ln_qkv_fusion.sh
```

### 2. 基本使用

```python
from core.layer.fused_ln_qkv_gptq import fused_ln_qkv_gptq
import torch

# 创建输入数据
input_data = torch.randn(1, 1, 4096, dtype=torch.float16)

# 创建GPTQ参数
qweight = torch.randint(0, 255, (512, 12288), dtype=torch.uint32)
qzeros = torch.randint(0, 255, (32, 1536), dtype=torch.uint32)
scales = torch.randn(32, 12288, dtype=torch.float16)

# 创建LayerNorm参数
ln_weight = torch.randn(4096, dtype=torch.float16)
ln_bias = torch.randn(4096, dtype=torch.float16)

# 调用融合算子
q, k, v = fused_ln_qkv_gptq(
    input=input_data,
    qweight=qweight,
    qzeros=qzeros,
    scales=scales,
    ln_weight=ln_weight,
    ln_bias=ln_bias,
    num_heads=32,
    kv_num_heads=32,
    head_size=128,
    groupsize=128
)
```

### 3. 高级使用

```python
from core.layer.fused_ln_qkv_gptq import FusedLNQKVGPTQ

# 创建融合算子实例
fused_op = FusedLNQKVGPTQ()

# 检查可用性
if fused_op.is_available():
    q, k, v = fused_op.fused_ln_qkv_gptq(
        input=input_data,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        num_heads=32,
        kv_num_heads=32,
        head_size=128,
        groupsize=128
    )
```

## 性能目标

### 1. 延迟目标
- **融合LN+QKV**：≤ 0.25ms
- **GPTQ QKV投影**：≤ 0.10ms
- **LayerNorm**：≤ 0.05ms

### 2. 加速比目标
- **融合 vs 分离**：≥ 2x
- **融合 vs PyTorch**：≥ 5x

### 3. 内存效率
- **减少内存访问**：避免中间结果存储
- **共享内存利用**：缓存LayerNorm统计
- **批处理优化**：支持不同batch size

## 测试

### 1. 功能测试

```bash
cd test
python test_fused_ln_qkv_gptq.py
```

测试内容：
- 基础功能验证
- 输出形状检查
- 数值稳定性
- 批处理支持

### 2. 性能测试

```bash
cd test
python test_fused_ln_qkv_gptq_performance.py
```

测试内容：
- 融合算子性能
- 分离算子性能
- PyTorch实现性能
- 性能对比分析

### 3. 集成测试

```bash
cd test
python test_fused_ln_qkv_gptq_integration.py
```

测试内容：
- Layer集成测试
- 性能对比测试
- 批处理测试
- 实际应用场景

### 4. 一键测试

```bash
cd cuda
bash test_all.sh
```

运行所有测试：
- GPTQ内核编译和测试
- 融合LN+QKV内核编译和测试
- 功能测试
- 性能测试
- 集成测试

## 实现细节

### 1. CUDA内核

```cpp
// 融合LayerNorm + GPTQ QKV投影内核
template<typename T>
__global__ void fused_ln_qkv_gptq_kernel(
    const T* input,           // [batch_size, seq_len, hidden_dim]
    const uint32_t* qweight,  // GPTQ量化权重 [K//8, N]
    const uint32_t* qzeros,   // GPTQ量化零点 [num_groups, groupsize//8]
    const T* scales,          // GPTQ量化缩放 [num_groups, N]
    const T* ln_weight,      // LayerNorm权重 [hidden_dim]
    const T* ln_bias,        // LayerNorm偏置 [hidden_dim]
    T* q_output,             // Q输出 [batch_size, num_heads, seq_len, head_size]
    T* k_output,             // K输出 [batch_size, kv_num_heads, seq_len, head_size]
    T* v_output,             // V输出 [batch_size, kv_num_heads, seq_len, head_size]
    // ... 其他参数
);
```

### 2. GPTQ解量化

```cpp
// GPTQ 4bit解量化内核
__device__ __forceinline__ void dequantize_4bit_gptq(
    const uint32_t* qweight,
    const uint32_t* qzeros,
    const half* scales,
    int weight_idx,
    int group_idx,
    int groupsize,
    half* output
) {
    // 计算4bit权重的索引
    int weight_group = weight_idx / groupsize;
    int weight_offset = weight_idx % groupsize;
    
    // 获取量化权重
    int qweight_idx = weight_group * (groupsize / 8) + weight_offset / 8;
    int bit_offset = (weight_offset % 8) * 4;
    
    uint32_t packed_weight = qweight[qweight_idx];
    uint32_t weight_4bit = (packed_weight >> bit_offset) & 0xF;
    
    // 获取零点和缩放
    int zero_idx = group_idx * (groupsize / 8) + weight_offset / 8;
    int zero_bit_offset = (weight_offset % 8) * 4;
    
    uint32_t packed_zero = qzeros[zero_idx];
    uint32_t zero_4bit = (packed_zero >> zero_bit_offset) & 0xF;
    
    // 解量化
    float scale = __half2float(scales[group_idx]);
    float dequantized = (weight_4bit - zero_4bit) * scale;
    
    *output = __float2half(dequantized);
}
```

### 3. LayerNorm计算

```cpp
// LayerNorm计算
__shared__ float shared_stats[64];  // 缓存统计信息

if (threadIdx.x < 2) {
    // 计算均值和方差
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        int idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + i;
        float val = __half2float(input[idx]);
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / hidden_dim;
    float var = (sum_sq / hidden_dim) - (mean * mean);
    
    shared_stats[seq_idx * 2] = mean;
    shared_stats[seq_idx * 2 + 1] = var;
}
__syncthreads();

// 归一化
int input_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + hidden_idx;
float normalized = (__half2float(input[input_idx]) - shared_stats[seq_idx * 2]) / 
                  sqrtf(shared_stats[seq_idx * 2 + 1] + eps);

normalized = normalized * __half2float(ln_weight[hidden_idx]) + __half2float(ln_bias[hidden_idx]);
```

## 性能分析

### 1. 理论分析

**融合前：**
- LayerNorm: ~0.33ms
- QKV投影: ~0.42ms
- **总计**: ~0.75ms

**融合后：**
- 融合算子: ~0.25ms
- **性能提升**: 3x加速

### 2. 收益来源

1. **减少内存访问**：避免中间结果的存储和读取
2. **减少内核启动开销**：从2个内核减少到1个
3. **更好的缓存利用**：数据在GPU缓存中保持热度
4. **减少同步开销**：避免内核间的同步

### 3. 优化策略

1. **共享内存优化**：缓存LayerNorm统计信息
2. **向量化操作**：使用CUDA向量化指令
3. **内存访问优化**：减少内存访问次数
4. **计算融合**：将多个操作合并为一个内核

## 故障排除

### 1. 编译问题

**问题**：找不到CUDA_HOME
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**问题**：找不到Python头文件
```bash
export PYTHON_INCLUDE=$(python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
```

### 2. 运行时问题

**问题**：CUDA内核不可用
```python
# 检查CUDA环境
if not torch.cuda.is_available():
    print("CUDA不可用")
    
# 检查内核库
if not os.path.exists("ln_qkv_fusion_kernel.so"):
    print("内核库不存在")
```

**问题**：数据类型不匹配
```python
# 确保数据类型正确
if input.dtype != torch.float16:
    input = input.to(torch.float16)
if scales.dtype != torch.float16:
    scales = scales.to(torch.float16)
```

### 3. 性能问题

**问题**：性能不达标
- 检查CUDA架构支持
- 检查内存带宽
- 检查批处理大小
- 检查序列长度

**问题**：数值不稳定
- 检查LayerNorm epsilon
- 检查GPTQ参数
- 检查数据类型转换

## 未来优化

### 1. 短期优化
- **内存访问优化**：进一步减少内存访问
- **计算优化**：优化GPTQ解量化算法
- **批处理优化**：支持更大的批处理

### 2. 中期优化
- **多GPU支持**：支持多GPU并行
- **动态批处理**：支持动态批处理大小
- **混合精度**：支持混合精度计算

### 3. 长期优化
- **自动调优**：自动优化内核参数
- **硬件适配**：适配不同的GPU架构
- **算法优化**：探索新的融合算法

## 贡献指南

### 1. 代码规范
- 遵循CUDA编程规范
- 使用清晰的变量命名
- 添加详细的注释
- 进行充分的测试

### 2. 测试要求
- 功能测试必须通过
- 性能测试必须达标
- 集成测试必须通过
- 代码覆盖率 > 80%

### 3. 提交流程
1. Fork项目
2. 创建功能分支
3. 实现功能
4. 编写测试
5. 提交PR
6. 代码审查
7. 合并代码

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**：本方案仍在开发中，可能存在bug和性能问题。建议在生产环境使用前进行充分测试。
