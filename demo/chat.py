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
import re
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


def _find_stop(full_text, fmt):
    """在 base 模型生成的 full_text 中定位停止边界。
    DeepSeek base 模型回答完会续写 "User: .../Assistant: ..." 伪对话，
    需在第一个行首的 "User:"/"Assistant:" 处截断，只保留真正回答。
    Qwen 格式在 <|im_end|> 处截断。
    返回 (截断后的回答, 是否截断)。"""
    if fmt == "qwen":
        idx = full_text.find("<|im_end|>")
        if idx != -1:
            return full_text[:idx], True
        return full_text, False
    # DeepSeek base: 行首的 "User:" 或 "Assistant:" 视为续写边界
    m = re.search(r"(?:^|\n)\s*(?:User:|Assistant:)", full_text)
    if m:
        return full_text[:m.start()].rstrip(), True
    return full_text, False


def _safe_print_len(text, fmt):
    """计算可安全打印的长度：末尾若可能是停止序列的前缀，则先扣留，等下个 chunk 确认。
    避免把 "\n\nUser" 打出来后才发现要截断（屏幕残留）。"""
    if fmt == "qwen":
        marker = "<|im_end|>"
        # 扣留可能是 "<|im_end|>" 前缀的尾部
        for k in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:k]):
                return len(text) - k
        return len(text)
    # DeepSeek: 停止边界是 (行首) \n\s* (User:|Assistant:)
    # 扣留尾部「\n + 空白(含换行) + (User|Assistant 的前缀)?」整段
    m = re.search(r"\n[ \t\r\n]*([A-Za-z]*)$", text)
    if m:
        partial = m.group(1)
        for kw in ("User", "Assistant"):
            if kw.startswith(partial) and partial:  # partial 是 kw 的前缀
                return m.start()
        if not partial:  # 只有 \n + 空白，仍可能是边界起始
            return m.start()
    return len(text)


# ========== 流式请求 ==========
def _stop_strings(fmt):
    """服务端停止字符串：命中即由 server 终止生成，避免 client 提前断流导致
    server 端 seq 孤儿（孤儿 seq 会与下个请求共用常驻 block_table/seqlens 缓冲 → 状态错乱）。

    注意：必须带换行前缀（\nUser: / \nAssistant:）。DeepSeek prompt 用 \n\n 分隔轮次，
    续写边界必在行首。若用裸 "User:"/"Assistant:"，模型生成代码/正文时一旦出现这
    两个字（变量名、注释等）就会被误判成续写边界、提前截断正常内容。"""
    if fmt == "qwen":
        return ["<|im_end|>"]
    return ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"]


def stream_chat(prompt, url, max_tokens, temperature, repetition_penalty, fmt):
    """对接 /generate_stream (SSE)。返回 (总耗时, 首字延迟, 截断后的回答)。

    服务端按 stop 串终止生成；client 侧仍用 _find_stop 做精确显示截断。
    实时打印时边累加边检测停止边界，避免把续写的 "User:/Assistant:" 打到屏幕。"""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stop": _stop_strings(fmt),
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    api = url.rstrip("/") + "/generate_stream"

    start_time = time.time()
    first_token_time = None
    buf = []          # 累计生成文本（未截断）
    printed_len = 0   # 已打印到屏幕的字符数（基于截断后的文本）
    hit_stop = False  # 是否触发了停止边界

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
                    buf.append(text)
                    truncated, hit = _find_stop("".join(buf), fmt)
                    # 只打印到「安全长度」——扣留可能是停止序列前缀的尾部
                    safe = _safe_print_len(truncated, fmt) if not hit else len(truncated)
                    new = truncated[printed_len:safe]
                    if new:
                        print(new, end="", flush=True)
                    printed_len = safe
                    if hit:
                        hit_stop = True
                        break  # 到达停止边界，丢弃后续续写
                if chunk.get("finished"):
                    break
            # 流结束时若仍有扣留的尾部（未触发停止边界），补打出来
            if not hit_stop:
                final_trunc, _ = _find_stop("".join(buf), fmt)
                tail = final_trunc[printed_len:]
                if tail:
                    print(tail, end="", flush=True)
            print("\n")
            total = time.time() - start_time
            first_lat = first_token_time - start_time if first_token_time else None
            reply, _ = _find_stop("".join(buf), fmt)
            return total, first_lat, reply
    except requests.exceptions.RequestException as e:
        print(f"\n[ERR] 请求异常: {e}", flush=True)
        return None, None, ""


# ========== 主循环 ==========
def main():
    p = argparse.ArgumentParser(description="micro-vllm 交互式聊天 demo")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="api_server 地址")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repetition-penalty", type=float, default=1.15,
                   help="repetition penalty（>1 惩罚已出现 token，缓解 base 模型重复；1.0 禁用）")
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
            prompt, args.url, args.max_tokens, args.temperature,
            args.repetition_penalty, args.format)

        if total is not None:
            if reply:
                history.append({"role": "assistant", "content": reply})
            lat = f"  |  首字: {first_lat:.2f}s" if first_lat else ""
            toks = len(reply)  # 近似 token 数（中文按字）
            tps = toks / total if total > 0 else 0
            print(f"⏱️  {total:.2f}s{lat}  |  ~{tps:.1f} tok/s  |  轮次:{(len(history)-1)//2}")


if __name__ == "__main__":
    main()
