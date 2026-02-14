import aiohttp
import asyncio


async def send_request(session, data):
    full_response = ""  # 用于存储完整的响应
    async with session.post(
            "http://localhost:8000/generate_stream",
            json=data,
            headers={"Content-Type": "application/json"}
    ) as response:
        async for chunk in response.content:
            chunk_str = chunk.decode('utf-8')  # 将字节转换为字符串
            full_response += chunk_str
            print(f"Received chunk for prompt '{data['prompt'][:20]}...': {chunk_str.strip()}")

    # 请求完成后打印完整响应
    print(f"\n完整响应 for prompt '{data['prompt'][:20]}...':\n{full_response}\n")
    return full_response


async def main():
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
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, data) for data in prompts]
        responses = await asyncio.gather(*tasks)

        # 如果需要，可以在这里处理所有响应
        # for i, response in enumerate(responses):
        #     print(f"\nResponse {i+1}:\n{response}")


asyncio.run(main())