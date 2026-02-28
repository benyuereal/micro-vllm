from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

# 必要修改 1：新增 device 参数（默认 None，兼容原有调用）
def load_model(model_path, device=None):
    try:
        # 你原有 tokenizer 逻辑完全不动
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

    # 核心逻辑开始
    if device is not None:
        # TP 场景：禁用 accelerate 自动分片，强制到指定设备
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,  # 关键：指定具体设备（如 cuda:1）
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        # 兼容你原有逻辑：没传 device 时仍用 auto
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
    return model, tokenizer