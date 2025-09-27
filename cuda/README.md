# 融合LN+QKV GPTQ CUDA内核

这个目录包含了融合LayerNorm + GPTQ QKV投影的CUDA内核实现。

## 文件结构

```
cuda/
├── gptq_ln_qkv_fusion_kernel.cu # 主要的GPTQ LN+QKV融合内核实现
├── gptq_cuda_kernel_vllm.cu     # vLLM GPTQ内核参考实现
├── compile_fusion.py            # GPTQ LN+QKV融合内核编译脚本
├── test_fusion.py               # 融合内核测试脚本（编译+功能+性能）
├── compile_vllm.py              # vLLM内核编译脚本
├── test_vllm_gptq.py            # vLLM内核测试脚本
└── README.md                    # 本文件
```

## 功能特性

- ✅ **完整的GPTQ INT4动态反量化**：包含qzeros处理
- ✅ **最优的LayerNorm实现**：使用Welford算法，数值稳定
- ✅ **QKV融合**：在同一个CUDA内核中执行LayerNorm和QKV投影
- ✅ **只使用float16**：不需要bfloat16转换
- ✅ **参考vLLM实现**：充分借鉴vLLM的优化策略

## 快速开始

### 编译GPTQ LN+QKV融合内核

```bash
cd cuda
python compile_fusion.py
```

### 测试融合内核

```bash
python test_fusion.py
```

### 测试vLLM内核

```bash
python test_vllm_gptq.py
```

## 测试内容

### 融合内核测试 (`test_fusion.py`)
- **编译测试**：验证内核能够正常编译
- **功能测试**：验证基本功能正确性
- **性能测试**：测量执行时间（目标：< 0.5ms）

### vLLM内核测试 (`test_vllm_gptq.py`)
- **功能测试**：验证vLLM GPTQ内核功能
- **性能测试**：测量vLLM内核执行时间

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