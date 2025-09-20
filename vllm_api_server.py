# fixed_api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import time
import json
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
import uvicorn

# 全局引擎实例
engine = None
app = FastAPI(title="vLLM API Server", version="0.1.0")


# 请求模型
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


# 响应模型
class GenerateResponse(BaseModel):
    text: str
    tokens: int
    time_ms: float


class BatchGenerateResponse(BaseModel):
    results: List[GenerateResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# 初始化vLLM引擎
@app.on_event("startup")
async def startup_event():
    global engine
    model_path = "/root/Qwen-7B-Chat"

    print(f"Loading model from {model_path}...")
    try:
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            disable_log_stats=True  # 减少日志输出
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise e


# 健康检查端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model="Qwen-7B-Chat",
        device="cuda"
    )


# 非流式生成端点
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # 生成请求 - 使用正确的方法
        results = await engine.generate(
            request.prompt,
            sampling_params,
            request_id=f"req_{time.time_ns()}"
        )

        # 提取结果
        generated_text = results[0].outputs[0].text
        tokens_count = len(results[0].outputs[0].token_ids)

        end_time = time.time()
        time_ms = (end_time - start_time) * 1000

        return GenerateResponse(
            text=generated_text,
            tokens=tokens_count,
            time_ms=time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# 修复的流式生成端点，确保max_tokens生效
@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.stream:
        return await generate_text(request)

    # 确保max_tokens至少为1
    max_tokens = max(1, request.max_tokens or 128)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=None  # 明确设置stop为None，避免默认停止词
    )

    request_id = f"stream_{time.time_ns()}"


    async def stream_results() -> AsyncGenerator[str, None]:
        """正确的流式生成实现，确保max_tokens生效"""
        full_text = ""
        token_count = 0
        start_time = time.time()
        last_token_time = start_time
        is_finished = False

        try:
            last_output = None
            async for output in engine.generate(request.prompt, sampling_params, request_id):
                last_output = output

                if output.outputs and output.outputs[0].text:
                    text = output.outputs[0].text

                    # 检查是否是新文本
                    if text != full_text:
                        new_text = text[len(full_text):]
                        full_text = text

                        # 统计token数量
                        if new_text.strip():  # 只统计有内容的token
                            token_count += 1
                            current_time = time.time()
                            elapsed_time = current_time - start_time

                            # 计算吞吐量
                            if elapsed_time > 0:
                                tokens_per_sec = token_count / elapsed_time
                            else:
                                tokens_per_sec = 0

                            # 计算当前token的生成时间
                            token_time = current_time - last_token_time
                            last_token_time = current_time

                            # 检查是否完成的条件
                            # 1. 达到最大token数
                            # 2. 输出有finish_reason
                            is_finished = (
                                    token_count >= max_tokens or
                                    (output.outputs[0].finish_reason is not None and
                                     output.outputs[0].finish_reason != "length")
                            )


                            data = {
                                "token": output.outputs[0].token_ids[-1] if output.outputs[0].token_ids else None,
                                "text": new_text,
                                "full_text": full_text,
                                "finished": is_finished
                            }

                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                            # 如果判断完成，则退出循环
                            if is_finished:
                                print(
                                    f"Generation completed: token_count={token_count}, max_tokens={max_tokens}, finish_reason={output.outputs[0].finish_reason}")
                                break

            # 最终检查：如果循环结束但未设置完成标志
            if last_output and not is_finished:
                if (last_output.outputs and last_output.outputs[0].finish_reason is not None):
                    is_finished = True
                    print(f"Generation completed based on final finish_reason: {last_output.outputs[0].finish_reason}")

        except Exception as e:
            print(f"Error during stream generation: {str(e)}")
            raise

        # 流式结束时打印最终统计信息
        end_time = time.time()
        total_time = end_time - start_time
        final_tokens_per_sec = token_count / total_time if total_time > 0 else 0

        print(f"\n=== Stream Completed ===")
        print(f"tokens generated: {token_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {final_tokens_per_sec:.2f} tokens/sec")
        if token_count > 0:
            print(f"Average latency per token: {(total_time * 1000 / token_count):.2f} ms/token")
        print(f"text length: {len(full_text)} characters")
        print("===================================")

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_results(), media_type="text/event-stream")


# 批量生成端点
@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate_text(request: BatchGenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 确保max_tokens至少为1
        max_tokens = max(1, request.max_tokens or 128)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=None
        )

        # 为每个提示创建任务
        tasks = []
        for i, prompt in enumerate(request.prompts):
            task = engine.generate(
                prompt,
                sampling_params,
                request_id=f"batch_{i}_{time.time_ns()}"
            )
            tasks.append(task)

        # 等待所有任务完成
        outputs_list = await asyncio.gather(*tasks)

        # 构建响应
        response_results = []
        for outputs in outputs_list:
            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)

            response_results.append(GenerateResponse(
                text=text,
                tokens=tokens,
                time_ms=0
            ))

        return BatchGenerateResponse(results=response_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)