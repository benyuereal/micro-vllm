# api_server.py
import argparse
import asyncio
import json
import os
import time
import uuid
# CUDA 显存分配器：expandable_segments 消除碎片（长输入 prefill 的 W8A16 反量化
# 临时张量 320-680MB 需要大块连续显存；固定段分配器下 reserved-unallocated 碎片
# 导致 3072 输入 OOM 0/4 失败）。须在 import torch 前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from queue import Queue
from typing import List, Optional, Union

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

# OpenAI 兼容：served model 名（--served-model-name，默认取模型路径 basename）
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "")
# spec decode 串行锁：generate_spec_decode 是单序列同步路径（不走 scheduler 循环），
# 多线程并发会同时 forward 同一 engine 模型 → 用锁串行化（单用户基准场景无影响）。
_spec_lock = asyncio.Lock()


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
# OpenAI 兼容数据模型（/v1/chat/completions、/v1/completions、/v1/models）
# ------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List, None] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    # 默认 None（OpenAI 语义：未指定→服务端默认），而非 0.7。spec 路径对 None 走 greedy
    # （与基准一致），仅当客户端显式指定非 0 temperature 才 400。
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[dict] = None
    n: int = 1

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[dict] = None
    ignore_eos: bool = False  # vllm bench serve 发此参数；True=遇 EOS 不停跑满 max_tokens
    n: int = 1


def _finish_reason(seq) -> str:
    """从 Sequence 推导 OpenAI finish_reason：满 max_tokens=length，否则 stop（EOS/stop 串）。"""
    if seq is None:
        return "stop"
    return "length" if len(seq.output_ids) >= seq.max_tokens else "stop"


def _norm_stop(stop) -> List[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _chat_to_prompt(messages) -> str:
    """messages → chat template 渲染后的 prompt 文本（走 tokenizer.apply_chat_template）。"""
    msgs = [{"role": m.role, "content": m.content or ""} for m in messages]
    return engine.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def _chat_chunk(model, idx, delta, finish=None):
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]}


def _completion_chunk(model, idx, text, finish=None):
    return {"id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": idx, "text": text, "logprobs": None, "finish_reason": finish}]}


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


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
        ctx, _done = engine.tp_receive_batch()
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
# OpenAI 兼容端点（与 vLLM 对齐：/v1/chat/completions、/v1/completions、/v1/models）
# spec 启用时单请求走 generate_spec_decode（DFlash2 draft-verify-accept，101.5 tok/s 路径）；
# 未启用走 continuous batching 的 add_request 循环。
# ------------------------------
@app.get("/v1/models")
async def list_models():
    if not engine:
        raise HTTPException(503, "Model not loaded")
    name = SERVED_MODEL_NAME or engine.adapter.model_type
    return {"object": "list", "data": [
        {"id": name, "object": "model", "created": int(time.time()), "owned_by": "micro-vllm"}]}


def _spec_enabled() -> bool:
    return engine is not None and engine.spec_decode_enabled


def _spec_check_greedy(req) -> None:
    """spec 路径（generate_spec_decode）是贪心专用：采样请求显式 400，不静默忽略。"""
    if _spec_enabled() and (req.temperature is not None and req.temperature != 0):
        raise HTTPException(400, "spec decode 路径当前仅支持 greedy（temperature=0），"
                                 "采样请用非 spec 服务实例")


async def _run_spec(prompt: str, max_tokens: int, ignore_eos: bool = False) -> dict:
    """spec 路径：单序列同步 generate_spec_decode（锁串行化，executor 跑避免阻塞事件循环）。"""
    async with _spec_lock:
        def _call():
            res = engine.generate_spec_decode(prompt, max_tokens, ignore_eos=ignore_eos)
            # 可观测：慢请求直接看这行（acceptance 低→自由推理文本→tok/s 掉，属模型行为非 bug）
            n = len(res["tokens"])
            print(f"[spec] {n} tok {res['time_s']:.2f}s = {res['tok_s']:.1f} tok/s "
                  f"acc={res['avg_acceptance']:.3f} steps={res['num_steps']} "
                  f"ptok={len(engine.tokenizer.encode(prompt, add_special_tokens=True))}",
                  flush=True)
            return res
        return await asyncio.get_event_loop().run_in_executor(None, _call)


async def _spec_stream(prompt, model, max_tokens, created, cid, prompt_tokens, kind,
                       include_usage, ignore_eos: bool = False):
    """spec 路径真流式 SSE：executor 线程跑 generate_spec_decode，on_tokens 回调经
    call_soon_threadsafe 把每步 accepted+bonus 一批 token 推进 asyncio.Queue，
    主协程逐批 yield SSE 帧（TTFT/ITL 真实反映 prefill/step 耗时）。
    kind: "completion"（choices[i].text）/ "chat"（choices[i].delta.content）。
    ignore_eos: 透传，True 时遇 EOS 不停。"""
    loop = asyncio.get_event_loop()
    q: "asyncio.Queue" = asyncio.Queue()

    def _cb(ids):  # executor 线程内调用
        loop.call_soon_threadsafe(q.put_nowait, ids)

    async def _run():
        async with _spec_lock:
            def _call():
                res = engine.generate_spec_decode(prompt, max_tokens, on_tokens=_cb,
                                                  ignore_eos=ignore_eos)
                n = len(res["tokens"])
                print(f"[spec] {n} tok {res['time_s']:.2f}s = {res['tok_s']:.1f} tok/s "
                      f"acc={res['avg_acceptance']:.3f} steps={res['num_steps']} "
                      f"ptok={prompt_tokens}", flush=True)
                return res
            return await loop.run_in_executor(None, _call)

    task = asyncio.create_task(_run())
    tok = engine.tokenizer

    def _frame(text, finish=None, usage=None):
        if kind == "chat":
            delta = {"content": text} if text else {}
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        else:
            obj = {"id": cid, "object": "text_completion", "created": created, "model": model,
                   "choices": [{"index": 0, "text": text, "logprobs": None,
                                "finish_reason": finish}]}
        if usage is not None:
            obj["usage"] = usage
        return obj

    async def gen():
        if kind == "chat":
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0,
                                     "delta": {"role": "assistant", "content": ""},
                                     "finish_reason": None}]})
        try:
            while True:
                try:
                    ids = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if task.done():
                        # generate 已返回：其内部 on_tokens 的 put 都先于 future 完成被调度，
                        # 队列必已排空；再兜底 drain 一次防竞态
                        while True:
                            try:
                                yield _sse(_frame(tok.decode(q.get_nowait(),
                                                             skip_special_tokens=True)))
                            except asyncio.QueueEmpty:
                                break
                        break
                    continue
                yield _sse(_frame(tok.decode(ids, skip_special_tokens=True)))
            res = await task
        except asyncio.CancelledError:  # 客户端断开：终止后台生成
            task.cancel()
            raise
        ids = res["tokens"]
        finish = "length" if len(ids) >= max_tokens else "stop"
        yield _sse(_frame("", finish, _usage(prompt_tokens, len(ids)) if include_usage else None))
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/completions")
async def v1_completions(req: CompletionRequest):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    _spec_check_greedy(req)
    model = req.model or (SERVED_MODEL_NAME or engine.adapter.model_type)
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
    stop = _norm_stop(req.stop)
    max_tokens = req.max_tokens or 128
    created = int(time.time())
    cid = f"cmpl-{uuid.uuid4().hex[:24]}"

    if _spec_enabled() and req.stream:
        # spec 路径真流式（vllm bench serve 走 /v1/completions stream:true）：
        # 多 prompt 流式无意义（bench 单 prompt），取首 prompt
        p = prompts[0]
        ptok = len(engine.tokenizer.encode(p, add_special_tokens=True))
        include_usage = bool((req.stream_options or {}).get("include_usage", False))
        return await _spec_stream(p, model, max_tokens, created, cid, ptok,
                                  "completion", include_usage, req.ignore_eos)

    if _spec_enabled():
        # spec 路径：逐 prompt 串行（单用户基准场景）
        choices = []
        total_prompt = 0
        total_completion = 0
        for i, p in enumerate(prompts):
            res = await _run_spec(p, max_tokens, ignore_eos=req.ignore_eos)
            ids = res["tokens"]
            total_prompt += len(engine.tokenizer.encode(p, add_special_tokens=True))
            total_completion += len(ids)
            choices.append({"index": i, "text": res["text"], "logprobs": None,
                            "finish_reason": "length" if len(ids) >= max_tokens else "stop"})
        return {"id": cid, "object": "text_completion", "created": created, "model": model,
                "choices": choices, "usage": _usage(total_prompt, total_completion)}

    # 非 spec：continuous batching。Sequence 对象在 add_request 后进 waiting_queue，
    # 同一实例后续被 in-place 更新（output_ids 增长），持引用即可推导 finish_reason。
    futs = []
    for p in prompts:
        # 注意：不能用 `req.temperature or 0.7`——temperature=0（greedy）会被 or 当成
        # 假值替换成 0.7。显式判 None。
        temp = 0.0 if req.temperature is None else req.temperature
        topp = 0.9 if req.top_p is None else req.top_p
        sid = engine.add_request(p, max_tokens, temperature=temp, top_p=topp, stop=stop)
        seq = next((s for s in engine.scheduler.waiting_queue if s.seq_id == sid), None)
        futs.append((p, seq, engine.new_completion_future(sid)))
    choices = []
    total_prompt = 0
    total_completion = 0
    for i, (p, seq, fut) in enumerate(futs):
        text = await fut
        n_out = len(seq.output_ids) if seq else 0
        total_prompt += len(engine.tokenizer.encode(p, add_special_tokens=True))
        total_completion += n_out
        choices.append({"index": i, "text": text, "logprobs": None,
                        "finish_reason": _finish_reason(seq)})
    return {"id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": choices, "usage": _usage(total_prompt, total_completion)}


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: ChatCompletionRequest):
    if not engine:
        raise HTTPException(503, "Model not loaded")
    model = req.model or (SERVED_MODEL_NAME or engine.adapter.model_type)
    prompt = _chat_to_prompt(req.messages)
    stop = _norm_stop(req.stop)
    max_tokens = req.max_tokens or 128
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    prompt_tokens = len(engine.tokenizer.encode(prompt, add_special_tokens=True))

    if req.stream and _spec_enabled():
        # spec 路径走真逐 token 流式（_chat_stream 是"生成完再分块"假流式，TTFT 虚高）
        include_usage = bool((req.stream_options or {}).get("include_usage", False))
        return await _spec_stream(prompt, model, max_tokens, created, cid,
                                  prompt_tokens, "chat", include_usage)

    if req.stream:
        return await _chat_stream(req, model, prompt, max_tokens, stop, created, cid, prompt_tokens)

    if _spec_enabled():
        res = await _run_spec(prompt, max_tokens)
        ids = res["tokens"]
        return {"id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": res["text"]},
                             "logprobs": None,
                             "finish_reason": "length" if len(ids) >= max_tokens else "stop"}],
                "usage": _usage(prompt_tokens, len(ids))}

    # 显式判 None（不能 `or`：temperature=0 会被当假值替换成 0.7）
    temp = 0.0 if req.temperature is None else req.temperature
    topp = 0.9 if req.top_p is None else req.top_p
    sid = engine.add_request(prompt, max_tokens, temperature=temp, top_p=topp, stop=stop)
    seq = next((s for s in engine.scheduler.waiting_queue if s.seq_id == sid), None)
    text = await engine.new_completion_future(sid)
    n_out = len(seq.output_ids) if seq else 0
    return {"id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": None, "finish_reason": _finish_reason(seq)}],
            "usage": _usage(prompt_tokens, n_out)}


async def _chat_stream(req, model, prompt, max_tokens, stop, created, cid, prompt_tokens):
    """OpenAI 流式：先生成完整文本（spec 或非 spec），再按 token 分块发 SSE。
    非真逐 token 流式（micro 的 spec 路径是同步 generate），但 SSE 帧格式与 vLLM 对齐，
    客户端可正常解析。基准对比主要用非流式，流式用于接口兼容性验证。"""
    if _spec_enabled():
        res = await _run_spec(prompt, max_tokens)
        text, ids = res["text"], res["tokens"]
    else:
        temp = 0.0 if req.temperature is None else req.temperature
        topp = 0.9 if req.top_p is None else req.top_p
        sid = engine.add_request(prompt, max_tokens, temperature=temp, top_p=topp, stop=stop)
        seq = next((s for s in engine.scheduler.waiting_queue if s.seq_id == sid), None)
        text = await engine.new_completion_future(sid)
        ids = seq.output_ids if seq else []
    finish = "length" if len(ids) >= max_tokens else "stop"

    async def gen():
        yield _sse(_chat_chunk(model, 0, {"role": "assistant", "content": ""}))
        for tok in engine.tokenizer.tokenize(text):
            yield _sse(_chat_chunk(model, 0, {"content": tok}))
            await asyncio.sleep(0.0)
        yield _sse(_chat_chunk(model, 0, {}, finish))
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": _usage(prompt_tokens, len(ids))})
    return StreamingResponse(gen(), media_type="text/event-stream")


# ------------------------------
# 极简启动逻辑
# ------------------------------
if __name__ == "__main__":
    setup()
    # 模型路径解析：--model / --model-name (CLI) > MODEL_NAME 环境变量 > config.py 默认
    # 既支持完整路径，也支持短名（如 Qwen-7B-Chat、deepseek）自动解析到本地模型根。
    default_path = getattr(Config.ModelConfig, "MODEL_PATH", None)
    model_path = get_model_path_from_cli(default=default_path)
    # spec decode 参数：--spec-decode --draft-model <path>（或环境变量 SPEC_DECODE=1 + DRAFT_MODEL）
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-decode", action="store_true", default=os.environ.get("SPEC_DECODE", "") == "1")
    ap.add_argument("--draft-model", default=os.environ.get("DRAFT_MODEL", ""))
    ap.add_argument("--num-spec-tokens", type=int, default=7)
    ap.add_argument("--max-batch-size", type=int,
                    default=int(os.environ.get("MAX_BATCH_SIZE", "512")),
                    help="KV manager 按 max_batch*2 强制下限 block 数，大模型 spec 模式调小防 OOM")
    ap.add_argument("--max-context-length", type=int,
                    default=int(os.environ.get("MAX_CONTEXT_LENGTH", "1024")),
                    help="序列长度上限（prompt+output），默认 1024；长输出基准调大")
    ap.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", ""))
    args, _ = ap.parse_known_args()
    if args.served_model_name:
        SERVED_MODEL_NAME = args.served_model_name
    print(f"Rank {get_rank()}: Loading model from {model_path}...")
    engine = InferenceEngine(model_path, max_batch_size=args.max_batch_size,
                             spec_decode=args.spec_decode,
                             draft_model_path=args.draft_model or None,
                             num_speculative_tokens=args.num_spec_tokens,
                             max_context_length=args.max_context_length)
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