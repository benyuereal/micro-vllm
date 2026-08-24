# api_server.py
import asyncio
import json
import os
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
from core.model_loader import get_model_path_from_cli
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
    repetition_penalty: float = 1.0
    stop: List[str] = []
    stream: bool = False

class BatchGenerateReq(BaseModel):
    prompts: List[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop: List[str] = []

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
            engine.tp_broadcast_waiting()
            await asyncio.sleep(0.0)
            continue

        ctx = BatchInferenceContext(len(batch), batch_type, batch)
        engine.tp_broadcast_batch(ctx)
        engine.step(ctx)
        await asyncio.sleep(0.0)  # GPU 正在执行 forward，asyncio 开销被覆盖
        engine.collect(ctx)

        engine.tp_broadcast_tokens(ctx)
        engine.update_sequences(ctx.sequences)


def non_rank0_inference_loop():
    print(f"Rank {get_rank()}: Inference loop started")
    while running:
        ctx = engine.tp_receive_batch()
        if ctx.batch_type == "waiting" or ctx.batch_size == 0:
            time.sleep(0.000)
            continue

        engine.step(ctx)
        seqs = engine.tp_receive_tokens(ctx)
        engine.update_sequences(seqs)


# ------------------------------
# 极简API端点
# ------------------------------
@app.get("/health", response_model=HealthResp)
async def health():
    if not engine:
        raise HTTPException(503, "Model not loaded")
    return HealthResp(
        model=engine.adapter.model_type,
        device=str(engine.device),
        running_sequences=len(engine.scheduler.running_sequences),
        waiting_sequences=len(engine.scheduler.waiting_queue)
    )


@app.post("/generate", response_model=GenerateResp)
async def generate(req: GenerateReq):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    start = time.time()
    # 走后台 rank0_inference_loop 的 continuous batching：add_request 入队，
    # await 完成 Future。多个并发 /generate 共享同一个 scheduler batch，
    # 而非各自跑 engine.generate() 串行循环。
    seq_id = engine.add_request(req.prompt, req.max_tokens,
                                temperature=req.temperature, top_p=req.top_p,
                                repetition_penalty=req.repetition_penalty, stop=req.stop)
    fut = engine.new_completion_future(seq_id)
    text = await fut
    return GenerateResp(text=text, tokens=len(text), time_ms=(time.time()-start)*1000)


@app.post("/batch_generate", response_model=BatchGenerateResp)
async def batch_generate(req: BatchGenerateReq):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    start = time.time()
    # 全部入队后统一 await：所有 prompt 进同一批 continuous batching。
    futs = []
    for p in req.prompts:
        sid = engine.add_request(p, req.max_tokens,
                                 temperature=req.temperature, top_p=req.top_p,
                                 repetition_penalty=req.repetition_penalty, stop=req.stop)
        futs.append((p, engine.new_completion_future(sid)))
    results = []
    for p, fut in futs:
        text = await fut
        results.append(GenerateResp(text=text, tokens=len(text), time_ms=0))
    return BatchGenerateResp(results=results)


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
            req.top_p,
            req.repetition_penalty,
            req.stop
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
                        "finished": (token == engine.eos_token_id)
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                if engine.scheduler.is_finished(seq_id):
                    # 推送最终的 finished 标记（覆盖 stop 串/EOS/max_tokens 三种结束情形），
                    # 让 client 可据此断流，无需自己猜停止边界。
                    yield f"data: {json.dumps({'token': -1, 'text': '', 'full_text': full_text, 'finished': True}, ensure_ascii=False)}\n\n"
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
    # 模型路径解析：--model / --model-name (CLI) > MODEL_NAME 环境变量 > config.py 默认
    # 既支持完整路径，也支持短名（如 Qwen-7B-Chat、deepseek）自动解析到本地模型根。
    default_path = getattr(Config.ModelConfig, "MODEL_PATH", None)
    model_path = get_model_path_from_cli(default=default_path)
    print(f"Rank {get_rank()}: Loading model from {model_path}...")
    engine = InferenceEngine(model_path)
    print(f"Rank {get_rank()}: Model loaded on {engine.device}")

    if rank0():
        @app.on_event("startup")
        async def _startup():
            asyncio.create_task(rank0_inference_loop())

        @app.on_event("shutdown")
        async def _shutdown():
            global running
            running = False

        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    else:
        non_rank0_inference_loop()