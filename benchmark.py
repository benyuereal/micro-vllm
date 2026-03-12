import aiohttp
import asyncio
import sys
import time


async def send_request(session, data):
    text = ""  # 用于存储完整的响应
    token_count = 0  # 该请求的token数
    start_time = time.time()  # 记录该请求开始时间

    async with session.post(
            "http://localhost:8000/generate_stream",
            json=data,
            headers={"Content-Type": "application/json"}
    ) as response:
        async for chunk in response.content:
            chunk_str = chunk.decode('utf-8')  # 将字节转换为字符串
            if len(chunk_str) > 10:
                text = chunk_str
            # print(f"Received chunk for prompt '{data['prompt'][:20]}...': {chunk_str.strip()}")
            # 统计token数
            if chunk_str.strip():
                token_count += 1

    # 计算该请求耗时
    request_duration = time.time() - start_time

    # 请求完成后打印完整响应
    print(f"\n完整响应 for prompt '{data['prompt']}...':\n{text}\n")
    return text, token_count, request_duration


async def main(batch_size: int = 32):
    prompts = [
        {"prompt": "写一个 SpringBoot 文件上传代码", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释区块链的共识机制", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用JavaScript实现一个Promise限流池", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于元宇宙未来的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习网络安全？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较SQL和NoSQL数据库的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于时间旅行的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用Rust实现一个简单的链表结构", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于远程工作利弊的分析", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何成为一名全栈开发者？", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较微服务和单体架构的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于人工智能觉醒的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用TypeScript实现一个类型安全的状态管理", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "解释什么是碳中和以及实现路径", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何系统地学习数据结构与算法？", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较Docker和虚拟机的区别", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于外星接触的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个Kotlin版本的Android MVVM架构示例", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释机器学习中的过拟合与欠拟合", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于赛博朋克的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用Python实现一个简单的神经网络", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于数字隐私保护的思考", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习产品设计？给出核心要点", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较RESTful API与GraphQL", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于末世求生的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个C++版本的智能指针实现", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释什么是量子纠缠及其应用", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于未来教育模式的畅想", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何高效地阅读技术文档？", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较GitFlow与Trunk-Based开发流程", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用Java实现一个简单的RPC框架", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "解释Transformer架构在NLP中的作用", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于开源精神的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何培养技术团队的创新氛围？", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较AWS与Azure云服务的核心差异", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于星际殖民的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个Swift版本的iOS网络请求封装", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释什么是DevOps及其最佳实践", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于程序员职业倦怠的思考", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何做好技术选型？给出方法论", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较TensorFlow与PyTorch的适用场景", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用Python实现一个简单的Web服务器", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "解释Kubernetes的核心概念与架构", "max_tokens": 500, "temperature": 0.6, "stream": True},
    ]

    # 根据 batch_size 截取 prompt 列表
    prompts_to_send = prompts[:batch_size]

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, data) for data in prompts_to_send]
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