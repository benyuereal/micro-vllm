# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import threading
import time
import json
from core.engine import InferenceEngine
import asyncio
from queue import Queue
import config.config as Config
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
    running_sequences: int
    waiting_sequences: int


# 初始化引擎
@app.on_event("startup")
async def startup_event():
    global engine
    model_path = Config.ModelConfig.MODEL_PATH  # fixme  修改为您的模型路径
    print(f"Loading model from {model_path}...")
    try:
        engine = InferenceEngine(model_path)
        print(f"Model loaded successfully on device: {engine.device}")
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
        model="Qwen-7B",
        device=str(engine.device),
        running_sequences=len(engine.scheduler.running_sequences),
        waiting_sequences=len(engine.scheduler.waiting_queue)
    )


# 非流式生成端点
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # 使用引擎的generate方法
        results = engine.generate(
            [request.prompt],
            max_tokens=request.max_tokens
        )

        # 提取结果
        text = ""
        for seq, result_text in results.items():
            text = result_text
            break  # 只取第一个结果

        end_time = time.time()
        time_ms = (end_time - start_time) * 1000

        return GenerateResponse(
            text=text,
            tokens=len(text),
            time_ms=time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# 批量生成端点
@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate_text(request: BatchGenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # 使用引擎的generate方法
        results = engine.generate(
            request.prompts,
            max_tokens=request.max_tokens
        )

        # 构建响应
        response_results = []
        for seq, text in results.items():
            response_results.append(GenerateResponse(
                text=text,
                tokens=len(text),
                time_ms=0  # 批量生成不单独计算时间
            ))

        return BatchGenerateResponse(results=response_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


# 在 generate_stream 函数中添加统计
@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.stream:
        response = await generate_text(request)
        return response

    # 添加统计变量
    start_time = time.time()
    token_count = 0

    async def event_generator():
        nonlocal token_count
        token_queue = Queue()
        full_text = ""

        seq_id = engine.add_request(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        def callback(token, text):
            nonlocal token_count
            token_count += 1
            token_queue.put((token, text))

        engine.register_stream_callback(seq_id, callback)

        try:
            step_count = 0
            total_step_time = 0
            first_batch_type = None
            first_batch_size = None
            
            while True:
                step_start = time.time()
                
                engine.step()
                step_time = time.time() - step_start
                
                step_count += 1
                total_step_time += step_time
                
                # 记录首批次的类型和大小
                if step_count == 1:
                    first_batch_type = "prefill" if engine.scheduler.waiting_queue else "decode"
                    first_batch_size = len(engine.scheduler.running_sequences)

                # 每 50 步打印统计
                if step_count % 50 == 0:
                    avg_step = total_step_time / step_count * 1000
                    print(f"📊 Steps {step_count-49}~{step_count}: avg={avg_step:.1f}ms/step, batch={len(engine.scheduler.running_sequences)}")

                # 消费所有已生成 token
                while not token_queue.empty():
                    token, text = token_queue.get()
                    full_text += text

                    data = {
                        "token": token,
                        "text": text,
                        "full_text": full_text,
                        "finished": token == engine.eos_token_id
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 检查序列是否结束
                running_seqs = [seq for seq in engine.scheduler.running_sequences if seq.seq_id == seq_id]
                if not running_seqs:
                    avg_step_time = total_step_time / step_count * 1000
                    print(f"[DONE] 共 {step_count} steps, 首批次: type={first_batch_type}, size={first_batch_size}, 平均每step: {avg_step_time:.1f}ms")
                    end_time = time.time()
                    gen_time = end_time - start_time
                    tokens_per_sec = token_count / gen_time if gen_time > 0 else 0

                    print(f"\nStream generated {token_count} tokens in {gen_time:.2f} seconds")
                    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
                    break

                # ✅ 关键：让出控制权，但不等待
                await asyncio.sleep(0)  # ← 原为 0.01，现在改为 0

        finally:
            engine.unregister_stream_callback(seq_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)