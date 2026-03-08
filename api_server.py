# api_server.py
import asyncio
import json
import threading
import time
from queue import Queue
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config.config as Config
from core.engine import InferenceEngine
from core.inference_context import BatchInferenceContext
from core.parallel_config import get_rank, rank0, setup
from core.sequence import Sequence

# 全局变量
engine: Optional[InferenceEngine] = None
running = True
app = FastAPI(title="vLLM API Server", version="0.1.0")


# ------------------------------
# 极简数据模型
# ------------------------------
class GenerateReq(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class BatchGenerateReq(BaseModel):
    prompts: List[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResp(BaseModel):
    text: str
    tokens: int
    time_ms: float

class BatchGenerateResp(BaseModel):
    results: List[GenerateResp]

class HealthResp(BaseModel):
    status: str = "healthy"
    model: str = "Qwen-7B"
    device: str
    running_sequences: int
    waiting_sequences: int


# ------------------------------
# 核心推理逻辑
# ------------------------------
async def rank0_inference_loop():
    print(f"Rank 0: Inference loop started")
    while running:
        batch, batch_type = engine.get_next_batch()
        
        if batch_type == "waiting" or not batch:
            BatchInferenceContext(0, "waiting").broadcast()
            await asyncio.sleep(0.000)
            continue
        
        ctx = BatchInferenceContext(len(batch), batch_type, batch)
        ctx.broadcast()
        engine.step(ctx)
        ctx.broadcast()
        engine.update_sequences(ctx.sequences)
        await asyncio.sleep(0.0)


def non_rank0_inference_loop():
    print(f"Rank {get_rank()}: Inference loop started")
    tokenizer = engine.tokenizer
    while running:
        ctx = BatchInferenceContext.receive(tokenizer)
        if ctx.batch_type == "waiting" or ctx.batch_size == 0:
            time.sleep(0.000)
            continue
        
        engine.step(ctx)
        ctx = BatchInferenceContext.receive(tokenizer)
        engine.update_sequences(ctx.sequences)


# ------------------------------
# 极简API端点
# ------------------------------
@app.get("/health", response_model=HealthResp)
async def health():
    if not engine:
        raise HTTPException(503, "Model not loaded")
    return HealthResp(
        device=str(engine.device),
        running_sequences=len(engine.scheduler.running_sequences),
        waiting_sequences=len(engine.scheduler.waiting_queue)
    )


@app.post("/generate", response_model=GenerateResp)
async def generate(req: GenerateReq):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    start = time.time()
    results = engine.generate([req.prompt], req.max_tokens)
    text = next(iter(results.values()))
    return GenerateResp(text=text, tokens=len(text), time_ms=(time.time()-start)*1000)


@app.post("/batch_generate", response_model=BatchGenerateResp)
async def batch_generate(req: BatchGenerateReq):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    results = engine.generate(req.prompts, req.max_tokens)
    return BatchGenerateResp(results=[
        GenerateResp(text=t, tokens=len(t), time_ms=0) for t in results.values()
    ])


@app.post("/generate_stream")
async def generate_stream(req: GenerateReq):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    if not req.stream:
        return await generate(req)
    
    start_time = time.time()
    token_count = 0

    async def event_generator():

        nonlocal token_count
        token_queue = Queue()
        full_text = ""  

        seq_id = engine.add_request(
            req.prompt,
            req.max_tokens,
            req.temperature,
            req.top_p
        )

        def callback(token, text):
            nonlocal token_count
            token_count += 1
            token_queue.put((token, text))

        engine.register_stream_callback(seq_id, callback)

        try:
            while True:
                while not token_queue.empty():
                    token, text = token_queue.get()
                    full_text += text  # ✅ 现在绝对不会报错了

                    data = {
                        "token": token,
                        "text": text,
                        "full_text": full_text,
                        "finished": token == engine.eos_token_id
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                if engine.scheduler.is_finished(seq_id):
                    end_time = time.time()
                    gen_time = end_time - start_time
                    tokens_per_sec = token_count / gen_time if gen_time > 0 else 0

                    print(f"\nStream generated {token_count} tokens in {gen_time:.2f} seconds")
                    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
                    break
                await asyncio.sleep(0.0)

        finally:
            engine.unregister_stream_callback(seq_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------------
# 极简启动逻辑
# ------------------------------
if __name__ == "__main__":
    setup()
    print(f"Rank {get_rank()}: Loading model from {Config.ModelConfig.MODEL_PATH}...")
    engine = InferenceEngine(Config.ModelConfig.MODEL_PATH)
    print(f"Rank {get_rank()}: Model loaded on {engine.device}")

    if rank0():
        @app.on_event("startup")
        async def _startup():
            asyncio.create_task(rank0_inference_loop())
        
        @app.on_event("shutdown")
        async def _shutdown():
            global running
            running = False
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        non_rank0_inference_loop()