# CUDA内核测试

## 目标
- **QKV投影**: 0.10ms
- **输出投影**: 0.10ms
- **性能要求**: 如果超出目标几倍就没有存在的必要

## 文件结构
```
cuda/
├── gptq_cuda_kernel.cu    # 优化CUDA内核源码
├── gptq_cuda.py          # Python接口
├── compile.py             # 编译脚本
├── test.py               # 性能测试
├── test_all.sh           # 一键测试
└── README.md             # 说明文档
```

## 使用方法

### 1. 一键测试
```bash
chmod +x test_all.sh
./test_all.sh
```

### 2. 分步测试
```bash
# 编译内核
python compile.py

# 测试性能
python test.py
```

## 性能目标
- ✅ **优秀**: < 0.10ms
- ✅ **良好**: < 0.20ms
- ⚠️ **需要优化**: < 0.50ms
- ❌ **不达标**: > 0.50ms

## 优化策略
1. **线程块优化**: 256个线程，每个处理4个元素
2. **内存访问优化**: 共享内存缓存，向量化加载
3. **计算优化**: 16个元素并行处理
4. **编译优化**: -O3, -use_fast_math, -Xptxas=-O3