import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from config import Config
from model_loader import load_model_and_tokenizer, prepare_model_for_inference
from cache_manager import KVCacheManager
from scheduler import Scheduler
from engine import InferenceEngine
from request import Request

# 创建FastAPI应用
app = FastAPI(title="Qwen vLLM API")

# 全局变量
model = None
tokenizer = None
kv_cache_manager = None
scheduler = None
engine = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9


class CompletionResponse(BaseModel):
    request_id: str
    text: str
    finished: bool


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型和组件"""
    global model, tokenizer, kv_cache_manager, scheduler, engine

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()
    model = prepare_model_for_inference(model)

    # 获取模型配置
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # 初始化KV缓存管理器
    kv_cache_manager = KVCacheManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim
    )

    # 初始化调度器
    scheduler = Scheduler(kv_cache_manager, max_batch_size=Config.MAX_BATCH_SIZE)

    # 初始化推理引擎
    engine = InferenceEngine(model, tokenizer, kv_cache_manager, scheduler)
    engine.start()

    print("Server started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    if engine:
        engine.stop()
    print("Server shutdown")


@app.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """创建文本补全请求"""
    # 创建请求对象
    req = Request.from_prompt(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    # 添加到调度器
    if not scheduler.add_request(req):
        raise HTTPException(status_code=503, detail="Server is busy, please try again later")

    # 等待请求完成（在实际实现中应该使用更高效的异步方式）
    while not req.finished:
        await asyncio.sleep(0.1)

    # 返回结果
    output_text = tokenizer.decode(req.output_tokens, skip_special_tokens=True)
    return CompletionResponse(
        request_id=req.request_id,
        text=output_text,
        finished=req.finished
    )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "queue_size": scheduler.get_queue_size()}


if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)