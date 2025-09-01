#!/usr/bin/env python3
"""
vLLM框架基本使用示例
"""

import torch
from vllm import LLMEngine, SamplingParams


def main():
    # 初始化引擎
    engine = LLMEngine(
        model="/data/model/qwen/Qwen-7B-Chat",
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_seq_length=2048,
        tokenizer="/data/model/qwen/Qwen-7B-Chat"
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=100,
    )

    # 准备提示
    prompts = [
        "hello, how are you?",
        "Hello, my name is",
    ]

    print("开始生成...")

    # 生成文本
    outputs = engine.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )

    # 打印结果
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {prompts[i]}")
        print(f"Generated: {output.generated_text}")
        print("---")

    # 关闭引擎
    engine.shutdown()


if __name__ == "__main__":
    main()