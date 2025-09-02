from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",  # 强制使用 CPU
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, tokenizer