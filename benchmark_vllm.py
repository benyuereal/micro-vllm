import aiohttp
import asyncio
import sys
import time
import json

# vLLM 服务地址
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen-7b-chat"

async def send_request(session, prompt, max_tokens=512):
    """发送流式请求，统计性能指标"""
    full_text = ""
    token_count = 0
    # 计时
    request_start = time.time()
    first_token_time = None  # 首token时间
    decode_start_time = None  # 生成开始时间

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }

    try:
        async with session.post(API_URL, json=payload, headers={"Content-Type": "application/json"}) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    # 解析 vLLM 流式数据
                    data = json.loads(line[6:])
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        token_count += 1
                        full_text += delta
                        # 记录首token时间
                        if first_token_time is None:
                            first_token_time = time.time() - request_start
                            decode_start_time = time.time()

        # 计算耗时
        total_duration = time.time() - request_start
        # decode 生成耗时（排除首token等待）
        decode_duration = time.time() - decode_start_time if decode_start_time else total_duration
        # decode 速度 = 生成token数 / 生成耗时
        decode_speed = token_count / decode_duration if decode_duration > 0 else 0

        # 打印单请求结果
        print(f"\n✅ 请求完成: {prompt[:30]}...")
        print(f"   生成token数: {token_count} | 首token时延: {first_token_time:.2f}s | decode速度: {decode_speed:.2f} tokens/s | 总耗时: {total_duration:.2f}s")

    except Exception as e:
        print(f"\n❌ 请求失败: {prompt[:30]}... | 错误: {str(e)}")
        return 0, 0, 0

    return token_count, total_duration, decode_speed

async def main(batch_size: int = 8):
    # 测试用 prompts
    test_prompts = [
        "写一个SpringBoot文件上传代码",
        "解释区块链的共识机制",
        "用JavaScript实现Promise限流池",
        "写一篇关于元宇宙未来的短文",
        "如何学习网络安全？给出学习路径",
        "比较SQL和NoSQL数据库的优缺点",
        "写一个关于时间旅行的科幻故事开头",
        "用Rust实现一个简单的链表结构",
        "写一篇关于远程工作利弊的分析",
        "如何成为一名全栈开发者？"
    ]

    # 截取批量大小
    prompts = test_prompts[:batch_size]
    print(f"🚀 启动 vLLM 并发测试 | 并发数: {batch_size} | 模型: {MODEL_NAME}")
    print("=" * 100)

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)

    # 汇总统计
    total_tokens = sum(r[0] for r in results)
    total_time = max(r[1] for r in results)  # 并发总耗时取最大值
    avg_decode_speed = sum(r[2] for r in results) / len([r for r in results if r[2]>0]) if total_tokens>0 else 0
    total_throughput = total_tokens / total_time if total_time>0 else 0

    print("\n" + "=" * 100)
    print(f"📊 测试汇总")
    print(f"并发请求数: {len(prompts)}")
    print(f"总生成token数: {total_tokens}")
    print(f"并发总耗时: {total_time:.2f}s")
    print(f"平均decode速度: {avg_decode_speed:.2f} tokens/s")
    print(f"整体吞吐量: {total_throughput:.2f} tokens/s")
    print("=" * 100)

if __name__ == "__main__":
    # 命令行传参：python test_vllm.py 8 （并发8）
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    asyncio.run(main(batch_size=batch_size))
