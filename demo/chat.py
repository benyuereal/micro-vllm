#!/usr/bin/env python3
"""
GLM-5.1 多轮交互式聊天 (流式，带上下文)
用法: python chat_test.py
"""

import requests
import json
import time
import sys

# ========== 配置 ==========
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "/models/GLM-5.1-Channel-INT4-w4a8"
MAX_TOKENS = 1500
TEMPERATURE = 0.7
SYSTEM_PROMPT = "你是一个有帮助的AI助手。"   # 可自定义系统提示

# ========== 流式请求函数（带历史） ==========
def stream_chat(messages):
    """
    发送流式请求，打印实时回复，返回 (总耗时, 首字延迟, 完整回复内容)
    messages: 完整的对话消息列表 [{"role": "...", "content": "..."}, ...]
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True
    }

    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    first_token_time = None
    full_response = []

    try:
        with requests.post(API_URL, json=payload, headers=headers, stream=True) as resp:
            if resp.status_code != 200:
                print(f"\n❌ HTTP 错误 {resp.status_code}: {resp.text}")
                return None, None, ""

            print("\n🤖 助手: ", end="", flush=True)

            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    break

                try:
                    chunk = json.loads(line.decode("utf-8"))
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        print(content, end="", flush=True)
                        full_response.append(content)
                except json.JSONDecodeError:
                    pass

            print("\n")
            total_time = time.time() - start_time
            first_latency = first_token_time - start_time if first_token_time else None
            return total_time, first_latency, "".join(full_response)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求异常: {e}")
        return None, None, ""

# ========== 主循环 ==========
def main():
    print("=" * 50)
    print("GLM-5.1 多轮交互式聊天 (带上下文)")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空历史重新开始")
    print("=" * 50)

    # 初始化对话历史（可包含系统提示）
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("✅ 对话历史已清空，重新开始。")
            continue
        if not user_input:
            continue

        # 将用户消息加入历史
        history.append({"role": "user", "content": user_input})

        # 发送请求并获取回复
        total_time, first_lat, assistant_reply = stream_chat(history)

        if total_time is not None:
            # 将助手回复追加到历史中（用于下一轮）
            if assistant_reply:
                history.append({"role": "assistant", "content": assistant_reply})
            latency_str = f"  |  首字延迟: {first_lat:.2f}s" if first_lat else ""
            print(f"⏱️  总耗时: {total_time:.2f}s{latency_str}")
            print(f"📝 当前对话轮次: {(len(history)-1)//2}")   # 不计 system 消息

if __name__ == "__main__":
    main()