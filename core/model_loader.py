import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM
from bitsandbytes import functional as bnb_F


def load_model(config_path: str):
    """加载量化模型并应用INT4-aware优化"""
    with open(config_path) as f:
        config = json.load(f)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        local_files_only=True
    )

    # 加载模型
    try:
        if config.get("use_quantization", False):
            if config["quantization_type"] == "bitsandbytes":
                # INT4量化配置
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.uint8,  # 优化存储
                )

                model = AutoModelForCausalLM.from_pretrained(
                    config["model_path"],
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16
                )

                # 应用INT4-aware优化
                _apply_int4_optimizations(model)

            elif config["quantization_type"] == "gptq":
                # GPTQ量化配置
                quantization_config = GPTQConfig(
                    bits=4,
                    group_size=128,
                    desc_act=False,
                    dtype=torch.bfloat16,
                    use_cuda_fp16=True,
                    disable_exllama=True
                )

                model = AutoGPTQForCausalLM.from_quantized(
                    config["model_path"],
                    device="cuda:0",
                    use_safetensors=True,
                    use_triton=False,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16
                )

                # 应用GPTQ INT4优化
                _apply_gptq_optimizations(model)
            else:
                raise ValueError(f"不支持的量化类型: {config['quantization_type']}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )

        print(f"模型加载成功: {config.get('quantization_type', '无量化')}")

    except Exception as e:
        print(f"模型加载错误: {e}")
        import traceback
        traceback.print_exc()

        # 回退方案
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            print("回退: 无量化模型加载成功")
        except Exception as e2:
            raise RuntimeError(f"模型加载失败: {e2}")

    return model, tokenizer


def _apply_int4_optimizations(model):
    """应用INT4-aware优化到bitsandbytes模型"""
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "device"):
            if module.weight.device.type == "cuda":
                # 替换前向传播为INT4-aware版本
                original_forward = module.forward
                module.forward = lambda x: bnb_F.linear4bit(x, module.weight, module.bias)
    print("应用INT4-aware优化到bitsandbytes模型")


def _apply_gptq_optimizations(model):
    """应用INT4-aware优化到GPTQ模型"""
    # 优化所有GPTQ线性层
    for name, module in model.named_modules():
        if hasattr(module, "qweight"):
            # 启用快速INT4计算路径
            module.optimize_weight_access = True
            module.use_cuda_fp16 = True
    print("应用INT4-aware优化到GPTQ模型")