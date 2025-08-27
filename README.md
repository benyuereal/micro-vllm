# vLLM Framework

一个高性能、内存高效的LLM推理引擎，支持Continuous Batching和Page Attention技术。

## 特性

- 🚀 **Continuous Batching**: 动态批处理，提高GPU利用率
- 💾 **Page Attention**: 分页注意力机制，减少内存碎片
- 🔥 **高性能**: 优化的CUDA内核，实现极致性能
- 🤖 **多模型支持**: 支持Qwen、LLaMA、ChatGLM等主流模型
- 📦 **生产就绪**: 提供API服务器和完整的管理工具

## 安装
```bash

pip install -r requirements.txt

python setup.py install
```
## 快速开始
```python

from vllm import LLMEngine, SamplingParams

初始化引擎
engine = LLMEngine(model="Qwen/Qwen-7B")

设置采样参数
sampling_params = SamplingParams(

temperature=0.8,

top_p=0.9,

max_tokens=100,

)

生成文本
outputs = engine.generate(

prompts=["Hello, my name is", "The future of AI is"],

sampling_params=sampling_params,

)

for output in outputs:

print(f"Prompt: {output.prompt}")

print(f"Generated text: {output.generated_text}")
```
## API服务器

启动API服务器：
```bash

python -m vllm.server
```
然后使用curl或客户端发送请求：
```bash

curl -X POST "http://localhost:8000/generate" \

-H "Content-Type: application/json" \

-d '{

"prompt": "Hello, my name is",

"sampling_params": {

"temperature": 0.8,

"max_tokens": 100

}

}'
```
## 性能基准

在NVIDIA A100上测试：

| 模型 | 吞吐量 (tokens/sec) | 延迟 (ms/token) |
|------|---------------------|-----------------|
| Qwen-7B | 1250 | 45 |
| LLaMA-13B | 890 | 62 |

## 文档

详细文档请参考 [文档链接]。

## 许可证

Apache License 2.0