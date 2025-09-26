# CUDA内核测试

## 目标
- **QKV投影**: 0.10ms
- **输出投影**: 0.10ms
- **性能要求**: 如果超出目标几倍就没有存在的必要

## 特性
- ✅ **cuBLAS集成**: 使用cuBLAS库加速
- ✅ **混合精度**: 支持FP16混合精度计算
- ✅ **Tensor Core**: 利用GPU的Tensor Core
- ✅ **共享内存**: 优化内存访问模式
- ✅ **向量化**: 64个元素并行处理

## 文件结构
```
cuda/
├── gptq_cuda_kernel.cu        # 当前版本CUDA内核
├── gptq_cuda_kernel_vllm.cu   # vLLM风格CUDA内核
├── gptq_cuda.py              # Python接口
├── compile.py                 # 编译脚本 (当前版本)
├── compile_vllm.py            # vLLM版本编译脚本
├── test.py                   # 性能测试 (当前版本)
├── test_vllm_only.py         # vLLM版本独立测试
├── test_vllm_comparison.py   # 性能对比测试
├── test_all.sh               # 一键测试
└── README.md                 # 说明文档
```

## 使用方法

### 1. 一键测试（推荐）
```bash
chmod +x test_all.sh
./test_all.sh
```

### 2. 分步测试
```bash
# 测试当前版本
python compile.py
python test.py

# 测试vLLM版本
python compile_vllm.py

# 性能对比测试
python test_vllm_comparison.py
```

## 性能目标
- ✅ **优秀**: < 0.10ms
- ✅ **良好**: < 0.20ms
- ⚠️ **需要优化**: < 0.50ms
- ❌ **不达标**: > 0.50ms

## 优化策略
1. **cuBLAS集成**: 使用cuBLAS库加速矩阵运算
2. **混合精度**: FP16混合精度计算
3. **Tensor Core**: 利用GPU的Tensor Core
4. **共享内存**: 1024字节共享内存缓存
5. **向量化**: 64个元素并行处理
6. **编译优化**: -O3, -use_fast_math, -Xptxas=-O3