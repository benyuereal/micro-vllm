from fastapi import FastAPI
from ..models.qwen import QwenEngine
from ..configs import qwen_7b as config

app = FastAPI()
engine = QwenEngine(config.MODEL_PATH)
engine.apply_optimizations()

@app.post("/generate")
async def generate(prompt: str):
    response = engine.prefill(prompt)
    return {"response": response}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}