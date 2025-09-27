# 融合LN+QKV GPTQ CUDA内核

这个目录包含了融合LayerNorm + GPTQ QKV投影的CUDA内核实现，目标是在0.3ms内完成完整的LayerNorm + GPTQ + QKV融合操作。

## 文件结构

```
cuda/
├── gptq_ln_qkv_fusion_kernel.cu # 主要的GPTQ LN+QKV融合内核实现
├── gptq_cuda_kernel_vllm.cu     # vLLM GPTQ内核参考实现（已修复）
├── compile_fusion.py            # GPTQ LN+QKV融合内核编译脚本
├── compile_vllm.py              # vLLM内核编译脚本
├── test_performance.py          # 融合内核性能测试脚本（目标0.25ms）
├── test_vllm_gptq.py            # vLLM内核测试脚本
└── README.md                    # 本文件
```

## 功能特性

- ✅ **完整的GPTQ INT4动态反量化**：包含qzeros处理，支持动态groupsize
- ✅ **最优的LayerNorm实现**：使用Welford算法，数值稳定，支持权重和偏置
- ✅ **QKV融合**：在同一个CUDA内核中执行LayerNorm和QKV投影
- ✅ **只使用float16**：不需要bfloat16转换，完全兼容
- ✅ **参考vLLM实现**：充分借鉴vLLM的优化策略和向量化技术
- ✅ **高性能优化**：目标延迟 < 0.3ms

## 快速开始

### 编译GPTQ LN+QKV融合内核

```bash
cd cuda
python compile_fusion.py
```

### 2. 性能测试（目标0.25ms，自动编译）

```bash
python test_performance.py
```

### 3. 编译vLLM内核（可选）

```bash
python compile_vllm.py
```

### 4. 测试vLLM内核（可选）

```bash
python test_vllm_gptq.py
```

## 测试内容

### 融合内核性能测试 (`test_performance.py`)
- **目标延迟**：< 0.25ms
- **功能验证**：输出形状、数值稳定性、非零检查
- **详细统计**：平均、最小、最大延迟和标准差
- **百分位数分析**：P50、P90、P95、P99延迟
- **目标评估**：验证是否达到0.25ms目标

### vLLM内核测试 (`test_vllm_gptq.py`)
- **功能测试**：验证vLLM GPTQ内核功能（已修复qzeros问题）
- **性能测试**：测量vLLM内核执行时间
- **对比分析**：与融合内核性能对比

## 内核参数

```cpp
torch::Tensor fused_ln_qkv_gptq_cuda(
    torch::Tensor input,           // [batch_size, seq_len, hidden_dim]
    torch::Tensor qweight,         // [hidden_dim//8, hidden_dim*3]
    torch::Tensor qzeros,          // [num_groups, groupsize//8]
    torch::Tensor scales,          // [num_groups, hidden_dim*3]
    torch::Tensor ln_weight,       // [hidden_dim]
    torch::Tensor ln_bias,         // [hidden_dim]
    int batch_size, int seq_len, int hidden_dim,
    int groupsize, float eps
) -> torch::Tensor  // 返回 [3, batch_size, seq_len, hidden_dim] 的QKV元组
```

## 性能优化

### 1. vLLM风格优化
- **分块策略**：`BLOCK_KN_SIZE=128`, `BLOCK_M_SIZE_MAX=8`
- **向量化技术**：`half2` 和 `__hfma2` 指令
- **共享内存**：`__shared__ half block_a[m_count][BLOCK_KN_SIZE]`
- **CUDA流**：`c10::cuda::getCurrentCUDAStream()`

### 2. 融合优化
- **单内核融合**：LayerNorm + GPTQ + QKV 在同一个内核中完成
- **内存优化**：减少中间张量创建和内存传输
- **计算优化**：使用原子操作避免竞争条件

### 3. 数值稳定性
- **Welford算法**：数值稳定的方差计算
- **GPTQ解量化**：`weight = (qweight - qzeros) * scale`
- **LayerNorm归一化**：`(x - mean) / sqrt(var + eps)`

## 性能目标

- **目标延迟**：< 0.25ms（相比0.5ms提升50%）
- **内存效率**：减少中间张量创建
- **数值精度**：与PyTorch实现完全一致
- **兼容性**：完全兼容现有PyTorch代码

## 依赖

- PyTorch >= 1.12
- CUDA >= 11.0
- Python >= 3.7

## 注意事项

- 确保CUDA环境正确配置
- 内核使用float16数据类型
- 支持动态groupsize
- 包含完整的GPTQ解量化逻辑
- 使用vLLM的PyTorch兼容性方法

## 使用建议

1. **首次使用**：先运行 `python compile_fusion.py` 验证编译
2. **性能测试**：运行 `python test_performance.py` 验证0.3ms目标
3. **对比测试**：运行 `python test_benchmark.py` 与vLLM对比
4. **集成使用**：参考 `test_fusion.py` 中的API调用方式