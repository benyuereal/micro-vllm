import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config


def load_model_and_tokenizer():
    """加载Qwen-7B-Chat模型和tokenizer"""
    print(f"Loading model from {Config.MODEL_PATH}...")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_PATH,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        torch_dtype=Config.DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    # 设置为评估模式
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def prepare_model_for_inference(model):
    """准备模型进行推理"""
    # 编译模型（如果支持）
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    # 添加模型状态检查
    if model is None:
        raise RuntimeError("Model is None after compilation")

    # 检查模型是否在GPU上
    if next(model.parameters()).device.type != "cuda":
        print("Warning: Model not on GPU!", "yellow")

    return model