from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_path):
    try:
        # 先尝试直接加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Fast tokenizer failed: {e}, trying slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

    # 配置4-bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4-bit量化
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
        bnb_4bit_use_double_quant=True,  # 使用双量化进一步节省内存
        bnb_4bit_quant_type="nf4",  # 量化类型为nf4
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",  # 自动分配设备
        trust_remote_code=True,
        local_files_only=True
    )
    return model, tokenizer