# LN+QKV融合内核测试使用说明

## 文件命名说明

| 文件名 | 功能描述 | 使用场景 |
|--------|----------|----------|
| `fusion.py` | 完整测试套件 | 首次测试、全面验证 |
| `quick.py` | 快速测试 | 开发调试、快速验证 |
| `layer_integration.py` | 层集成测试 | 集成验证、端到端测试 |
| `performance.py` | 性能测试 | 性能分析、基准测试 |

## 快速使用

### 1. 完整测试（推荐首次运行）
```bash
python fusion.py
```

### 2. 快速验证
```bash
python quick.py
```

### 3. 集成测试
```bash
python layer_integration.py
```

### 4. 性能测试
```bash
python performance.py
```

## 测试内容

- **fusion.py**: 编译 + 功能 + 精度 + 性能
- **quick.py**: 编译 + 功能 + 性能（简化版）
- **layer_integration.py**: 融合内核 + 层集成 + 形状验证
- **performance.py**: 性能分析 + 目标验证（0.25ms）

## 注意事项

- 所有测试都会自动编译内核（如果未编译）
- 目标延迟：< 0.25ms
- 支持完整的GPTQ解量化（包含qzeros处理）
- 使用float16数据类型
