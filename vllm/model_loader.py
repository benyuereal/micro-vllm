import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config


def load_model_and_tokenizer():
    """加载Qwen-7B-Chat模型和tokenizer"""
    print(f"Loading model from {Config.MODEL_PATH}...")

    # 确保使用正确的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"  # 确保左填充
    )
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token

    # 加载模型 - 确保使用正确配置
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        torch_dtype=Config.DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def prepare_model_for_inference(model):
    """准备模型进行推理（可选编译）"""
    # 临时禁用编译以测试是否导致问题
    if Config.ENABLE_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    return model