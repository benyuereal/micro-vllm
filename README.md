# vLLM Framework

一个高性能、内存高效的LLM推理引擎，支持Continuous Batching和Dynamic Cache技术。

## 特性

- 🚀 **Continuous Batching**: 动态批处理，提高GPU利用率
- 💾 **Dynamic Cache**: 动态KV缓存管理，减少内存使用
- 🔥 **高性能**: 优化的推理引擎，实现极致性能
- 🤖 **多模型支持**: 支持Qwen等主流模型
- 📦 **生产就绪**: 提供API服务器和完整的管理工具
- 🌊 **流式输出**: 支持实时流式文本生成

## 安装

```bash
pip install -r requirements.txt
```
## 快速开始
1. 基本使用
```python
from core.engine import InferenceEngine

# 初始化引擎
engine = InferenceEngine(model_path="/path/to/your/model")

# 生成文本
results = engine.generate([
    "Hello, my name is",
    "The future of AI is"
], max_tokens=100)

for seq, text in results.items():
    print(f"Prompt: {seq.prompt}")
    print(f"Generated text: {text}")
 ```
2. 流式生成
```python
from core.engine import InferenceEngine

# 初始化引擎
engine = InferenceEngine(model_path="/path/to/your/model")

# 流式生成
for token, text in engine.stream_generate("人工智能的未来是", max_tokens=50):
    print(text, end="", flush=True)
```
## API 服务器
启动API服务器：
```bash
python api_server.py
```
API 端点
健康检查
```bash
curl "http://localhost:8000/health"
```
非流式生成
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, my name is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
  ```
流式生成
```bash
curl -X POST "http://localhost:8000/generate_stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一个java版本的文件上传代码",
    "max_tokens": 1000,
    "temperature": 0.7,
    "stream": true
  }'
  ```
批量生成
```bash
curl -X POST "http://localhost:8000/batch_generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "写一首诗",
      "写一段java版本的文件上传代码"
    ],
    "max_tokens": 500
  }'
```
