#!/usr/bin/env python3
"""
micro-vllm 交互式聊天 demo（流式，带上下文）。

对接 micro-vllm 的 api_server `/generate_stream` (SSE) 接口。
支持多轮：把历史按模型 prompt 格式拼成单字符串发给 server。

用法:
  python demo/chat.py                                 # 默认 127.0.0.1:8000
  python demo/chat.py --url http://127.0.0.1:8001     # 指定 server
  python demo/chat.py --max-tokens 512 --temperature 0.7

DeepSeek-V2-Lite 是 base 模型，prompt 格式 "User: ...\n\nAssistant:"；
Qwen-Chat 用 <|im_start|>...<|im_end|>。本 demo 默认 DeepSeek 格式，
可用 --format qwen 切换。
"""

import argparse
import json
import sys
import time

import requests


# ========== prompt 格式 ==========
def build_prompt(history, fmt):
    """把对话历史拼成单字符串 prompt。history: [{"role","content"}, ...]（含 system）。"""
    if fmt == "qwen":
        parts = []
        for m in history:
            if m["role"] == "system":
                parts.append(f"<|im_start|>system\n{m['content']}<|im_end|>")
            elif m["role"] == "user":
                parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>")
            elif m["role"] == "assistant":
                parts.append(f"<|im_start|>assistant\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    else:
        # DeepSeek base 模型格式
        parts = []
        for m in history:
            if m["role"] == "system":
                parts.append(m["content"])
            elif m["role"] == "user":
                parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"Assistant: {m['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


def extract_assistant_reply(full_text, prompt, fmt):
    """server 返回的是从头开始的 full_text（含 prompt 前缀？否——/generate_stream
    的 full_text 只是生成部分）。我们的 /generate_stream 每个 chunk 的 text 是该 token
    的增量，full_text 是累计生成内容（不含 prompt），故直接用 full_text。"""
    return full_text


# ========== 流式请求 ==========
def stream_chat(prompt, url, max_tokens, temperature):
    """对接 /generate_stream (SSE)。返回 (总耗时, 首字延迟, 完整生成文本)。"""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    api = url.rstrip("/") + "/generate_stream"

    start_time = time.time()
    first_token_time = None
    full_response = []

    try:
        with requests.post(api, json=payload, headers=headers, stream=True) as resp:
            if resp.status_code != 200:
                print(f"\n[ERR] HTTP {resp.status_code}: {resp.text}", flush=True)
                return None, None, ""
            print("\n🤖 Assistant: ", end="", flush=True)
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    break
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                text = chunk.get("text", "")
                if text:
                    if first_token_time is None:
                        first_token_time = time.time()
                    print(text, end="", flush=True)
                    full_response.append(text)
                if chunk.get("finished"):
                    break
            print("\n")
            total = time.time() - start_time
            first_lat = first_token_time - start_time if first_token_time else None
            return total, first_lat, "".join(full_response)
    except requests.exceptions.RequestException as e:
        print(f"\n[ERR] 请求异常: {e}", flush=True)
        return None, None, ""


# ========== 主循环 ==========
def main():
    p = argparse.ArgumentParser(description="micro-vllm 交互式聊天 demo")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="api_server 地址")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--format", choices=["deepseek", "qwen"], default="deepseek",
                   help="prompt 格式（模型类型）")
    p.add_argument("--system", default="你是一个有帮助的AI助手。", help="系统提示")
    args = p.parse_args()

    print("=" * 56)
    print(f"micro-vllm 交互式聊天  (server={args.url}, format={args.format})")
    print("输入 quit/exit 退出 | clear 清空历史")
    print("=" * 56)

    history = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": args.system}]
            print("✅ 对话历史已清空。")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(history, args.format)

        total, first_lat, reply = stream_chat(
            prompt, args.url, args.max_tokens, args.temperature)

        if total is not None:
            if reply:
                history.append({"role": "assistant", "content": reply})
            lat = f"  |  首字: {first_lat:.2f}s" if first_lat else ""
            toks = len(reply)  # 近似 token 数（中文按字）
            tps = toks / total if total > 0 else 0
            print(f"⏱️  {total:.2f}s{lat}  |  ~{tps:.1f} tok/s  |  轮次:{(len(history)-1)//2}")


if __name__ == "__main__":
    main()
