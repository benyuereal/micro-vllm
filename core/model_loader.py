from transformers import AutoTokenizer, AutoModelForCausalLM
import json
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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    return model, tokenizer