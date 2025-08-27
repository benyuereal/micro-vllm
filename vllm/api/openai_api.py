import time
import uuid
from typing import List, Optional
from fastapi import HTTPException
from .models import (
    ChatCompletionRequest, CompletionRequest,
    ChatCompletionResponse, CompletionResponse,
    Message, Role, UsageInfo
)
from vllm import LLMEngine, SamplingParams


class OpenAIAPI:
    """OpenAI兼容的API实现"""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.model_name = "qwen-7b"  # 可以根据实际情况调整

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """创建聊天补全"""
        # 将消息转换为提示文本
        prompt = self._format_chat_prompt(request.messages)

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )

        # 生成响应
        responses = self.engine.generate(
            prompts=[prompt],
            sampling_params=sampling_params
        )

        # 提取生成的文本
        generated_text = responses[0].generated_text if responses else ""

        # 创建响应
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": Role.ASSISTANT,
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage=UsageInfo(
                prompt_tokens=0,  # 实际实现中需要计算token数量
                completion_tokens=0,
                total_tokens=0
            )
        )

    async def create_completion(self, request: CompletionRequest) -> CompletionResponse:
        """创建文本补全"""
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop
        )

        # 生成响应
        responses = self.engine.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params
        )

        # 提取生成的文本
        generated_text = responses[0].generated_text if responses else ""

        # 创建响应
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "text": generated_text,
                    "finish_reason": "stop"
                }
            ],
            usage=UsageInfo(
                prompt_tokens=0,  # 实际实现中需要计算token数量
                completion_tokens=0,
                total_tokens=0
            )
        )

    def _format_chat_prompt(self, messages: List[Message]) -> str:
        """将聊天消息格式化为Qwen模型所需的提示格式"""
        prompt = ""

        for message in messages:
            if message.role == Role.SYSTEM:
                prompt += f"System: {message.content}\n\n"
            elif message.role == Role.USER:
                prompt += f"User: {message.content}\n\n"
            elif message.role == Role.ASSISTANT:
                prompt += f"Assistant: {message.content}\n\n"

        # 添加最后的Assistant提示
        prompt += "Assistant:"

        return prompt