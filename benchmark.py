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
        {"prompt": "写一个java版本的文件上传代码", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个java版本的文件上传代码", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个java版本的文件上传代码", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一个关于太空探索的科幻故事开头", "max_tokens": 500, "temperature": 0.9, "stream": True},
        {"prompt": "写一个java版本的文件上传代码", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "写一篇关于人工智能伦理的短文", "max_tokens": 500, "temperature": 0.8, "stream": True},
        {"prompt": "如何学习深度学习？给出学习路径", "max_tokens": 500, "temperature": 0.7, "stream": True},
        {"prompt": "比较React和Vue框架的优缺点", "max_tokens": 500, "temperature": 0.6, "stream": True},
        {"prompt": "用Python实现快速排序算法", "max_tokens": 500, "temperature": 0.5, "stream": True},
        {"prompt": "解释量子计算的基本原理", "max_tokens": 500, "temperature": 0.6, "stream": True},
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