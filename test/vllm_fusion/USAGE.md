# vLLM融合内核测试使用说明

## 文件命名说明

| 文件名 | 功能描述 | 使用场景 |
|--------|----------|----------|
| `fusion.py` | 完整功能测试套件 | 首次测试、全面验证 |
| `quick.py` | 快速测试 | 开发调试、快速验证 |
| `performance.py` | 性能测试 | 性能分析、基准测试 |
| `integration.py` | 集成测试 | 集成验证、端到端测试 |

## 快速使用

### 1. 功能测试（推荐首次运行）
```bash
python fusion.py
```

### 2. 快速验证
```bash
python quick.py
```

### 3. 性能测试
```bash
python performance.py
```

### 4. 集成测试
```bash
python integration.py
```

## 测试内容

- **fusion.py**: 编译 + 功能 + 精度 + 性能
- **quick.py**: 快速验证（小矩阵尺寸）
- **performance.py**: 性能分析 + 基准测试
- **integration.py**: vLLM内核集成 + GPTQCUDAFusion集成

## 注意事项

- 所有测试都会自动编译内核（如果未编译）
- 目标延迟：< 0.3ms
- 支持完整的GPTQ解量化（包含qzeros处理）
- 使用float16数据类型
- 测试vLLM内核的修复版本
