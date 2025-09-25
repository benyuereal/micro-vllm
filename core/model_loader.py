import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from auto_gptq import AutoGPTQForCausalLM, GPTQConfig  # 添加AutoGPTQ支持


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

    # 加载模型
    try:
        if config.get("use_quantization", False):
            if config["quantization_type"] == "bitsandbytes":
                # BitsAndBytes量化配置
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    config["model_path"],
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16
                )
            elif config["quantization_type"] == "gptq":
                # GPTQ量化配置 - 优化参数设置
                quantization_config = GPTQConfig(
                    bits=4,  # 4-bit量化
                    group_size=128,  # 推荐组大小
                    desc_act=False,  # 禁用描述激活
                    dtype=torch.bfloat16  # 计算精度
                )

                # 使用AutoGPTQForCausalLM加载GPTQ模型
                model = AutoGPTQForCausalLM.from_quantized(
                    config["model_path"],
                    device="cuda:0",  # 指定GPU设备
                    use_safetensors=True,  # 使用safetensors格式
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16
                )
            else:
                raise ValueError(f"Unsupported quantization type: {config['quantization_type']}")
        else:
            # 不使用量化 - 标准加载方式
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )

        print(f"Model loaded successfully with {config.get('quantization_type', 'no quantization')}")

    except Exception as e:
        print(f"Error loading model: {e}")
        # 详细错误信息
        import traceback
        traceback.print_exc()

        # 回退方案：尝试不使用量化配置
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            print("Fallback: Model loaded without quantization")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model even without quantization: {e2}")

    return model, tokenizer