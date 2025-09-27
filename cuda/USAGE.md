# CUDA内核使用说明

## 快速开始

### 1. 编译融合内核
```bash
cd cuda
python compile_fusion.py
```

### 2. 运行性能测试（自动编译）
```bash
python test_performance.py
```

## 文件说明

- `gptq_ln_qkv_fusion_kernel.cu` - 融合内核源码（LayerNorm + GPTQ + QKV）
- `gptq_cuda_kernel_vllm.cu` - vLLM内核源码（已修复GPTQ解量化问题）
- `compile_fusion.py` - 融合内核编译脚本
- `compile_vllm.py` - vLLM内核编译脚本
- `test_performance.py` - 融合内核性能测试（自动编译）
- `test_vllm_gptq.py` - vLLM内核测试

## 注意事项

1. **自动编译**：`test_performance.py` 会自动编译内核（如果未编译）
2. **目标延迟**：融合内核目标 < 0.25ms
3. **功能完整**：两个内核都支持完整的GPTQ解量化（包含qzeros处理）

## 性能对比

- **融合内核**：LayerNorm + GPTQ + QKV 一体化，目标0.25ms
- **vLLM内核**：纯GPTQ实现，功能正确，性能优秀
