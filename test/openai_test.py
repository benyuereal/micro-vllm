import openai

# 配置客户端指向您的服务器
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy-key"  # 不需要真实密钥

# 聊天补全
response = openai.ChatCompletion.create(
    model="qwen-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)

# 文本补全
response = openai.Completion.create(
    model="qwen-7b",
    prompt="Once upon a time"
)

print(response.choices[0].text)