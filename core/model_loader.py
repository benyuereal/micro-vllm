import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig
import torch

def load_model(config_path):
    with open(config_path) as f:
        config = json.load(f)

    # 加载tokenizer (保持不变)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Fast tokenizer failed: {e}, trying slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

    quantization_config = None
    if config.get("use_quantization", False):
        if config["quantization_type"] == "bitsandbytes":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config["quantization_type"] == "gptq":
            quantization_config = GPTQConfig(
                bits=4,
                group_size=128,
                desc_act=False,
                dtype=torch.bfloat16
            )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16  # 这里明确指定数据类型
    )
    return model, tokenizer


# 使用示例
# model, tokenizer = load_model_from_config("config.json")