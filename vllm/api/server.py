import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .openai_api import OpenAIAPI
from .models import ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, CompletionResponse
from vllm import LLMEngine


class APIServer:
    """API服务器类"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="vLLM API Server", version="0.1.0")
        self.engine = None
        self.api = None

        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""

        @self.app.get("/")
        async def root():
            return {"message": "vLLM API Server is running"}

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "qwen-7b",
                        "object": "model",
                        "created": 1677610602,
                        "owned_by": "vllm"
                    }
                ]
            }

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            if not self.api:
                raise HTTPException(status_code=503, detail="API not initialized")
            return await self.api.create_chat_completion(request)

        @self.app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest):
            if not self.api:
                raise HTTPException(status_code=503, detail="API not initialized")
            return await self.api.create_completion(request)

    def initialize(self, model_path: str, **kwargs):
        """初始化模型和API"""
        print(f"Initializing model from: {model_path}")

        # 初始化引擎
        self.engine = LLMEngine(
            model=model_path,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            max_num_seqs=kwargs.get("max_num_seqs", 32),
            max_seq_length=kwargs.get("max_seq_length", 2048)
        )

        # 初始化API
        self.api = OpenAIAPI(self.engine)

        print("Model initialized successfully")

    def run(self):
        """运行服务器"""
        uvicorn.run(self.app, host=self.host, port=self.port)