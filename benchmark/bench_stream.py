"""HTTP 并发流式 /generate_stream 压测：N 条并发流式请求打 micro-vllm API 服务。

测什么：
  - 通过 aiohttp 并发调用 micro-vllm 服务的 /generate_stream 流式接口
  - N 条请求（固定 prompt 池循环取用），统计总 token 数与吞吐 tokens/s

用法：
  python3 bench_stream.py [batch_size]
  例: python3 bench_stream.py 32

依赖：
  - 已启动的 micro-vllm API 服务（python api_server.py，默认 http://localhost:8000，
    可用环境变量 API_URL 覆盖）
  - aiohttp 库
"""
import aiohttp
import asyncio
import os
import sys
import time

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# 固定 prompt 池（32 条），按 batch_size 循环取用
PROMPT_POOL = [
    "写一个 SpringBoot 文件上传代码",
    "解释区块链的共识机制",
    "用JavaScript实现一个Promise限流池",
    "写一篇关于元宇宙未来的短文",
    "如何学习网络安全？给出学习路径",
    "比较SQL和NoSQL数据库的优缺点",
    "写一个关于时间旅行的科幻故事开头",
    "用Rust实现一个简单的链表结构",
    "写一篇关于远程工作利弊的分析",
    "如何成为一名全栈开发者？",
    "比较微服务和单体架构的优缺点",
    "解释机器学习中的过拟合与欠拟合",
    "用Python实现一个简单的神经网络",
    "如何系统地学习数据结构与算法？",
    "解释什么是碳中和以及实现路径",
    "比较Docker和虚拟机的区别",
    "写一个Kotlin版本的Android MVVM架构示例",
    "用TypeScript实现一个类型安全的状态管理",
    "写一篇关于数字隐私保护的思考",
    "如何学习产品设计？给出核心要点",
    "比较RESTful API与GraphQL",
    "写一个C++版本的智能指针实现",
    "解释什么是量子纠缠及其应用",
    "写一篇关于未来教育模式的畅想",
    "如何高效地阅读技术文档？",
    "比较GitFlow与Trunk-Based开发流程",
    "用Java实现一个简单的RPC框架",
    "解释Transformer架构在NLP中的作用",
    "写一篇关于开源精神的短文",
    "如何培养技术团队的创新氛围？",
    "比较AWS与Azure云服务的核心差异",
    "解释Kubernetes的核心概念与架构",
]


async def send_request(session, data):
    text = ""  # 用于存储完整的响应
    token_count = 0  # 该请求的token数
    start_time = time.time()  # 记录该请求开始时间

    async with session.post(
            API_URL + "/generate_stream",
            json=data,
            headers={"Content-Type": "application/json"}
    ) as response:
        async for chunk in response.content:
            chunk_str = chunk.decode('utf-8')  # 将字节转换为字符串
            if len(chunk_str) > len(data['prompt']) + 5:
                text = chunk_str
            # 统计token数
            if chunk_str.strip():
                token_count += 1

    # 计算该请求耗时
    request_duration = time.time() - start_time

    # 请求完成后打印完整响应
    print(f"\n完整响应 for prompt '{data['prompt']}...':\n{text}\n")
    return text, token_count, request_duration


async def main(batch_size: int = 32):
    # 根据 batch_size 从 prompt 池循环取用
    prompts = [
        {"prompt": PROMPT_POOL[i % len(PROMPT_POOL)], "max_tokens": 500,
         "temperature": 0.7, "stream": True}
        for i in range(batch_size)
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, data) for data in prompts]
        results = await asyncio.gather(*tasks)

        # 计算总token数和总请求耗时
        total_tokens = sum(token_count for _, token_count, _ in results)
        total_time = sum(duration for _, _, duration in results)
        throughput = total_tokens / total_time if total_time > 0 else 0

        print("=" * 80)
        print(f"总请求数 : {len(tasks)}  总Token数: {total_tokens}, 总请求耗时: {total_time:.2f}秒, 吞吐率: {throughput:.2f} tokens/秒")
        print("=" * 80)


if __name__ == "__main__":
    # 默认 batch_size 为 32，可以通过命令行参数覆盖
    b_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    asyncio.run(main(batch_size=b_size))
