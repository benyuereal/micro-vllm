import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig
import torch
from auto_gptq import AutoGPTQForCausalLM

from weight_rearrange import rearrange_qwen_weights


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
                # BitsAndBytes量化配置 (保持不变)
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
                # GPTQ量化配置 - 专为Qwen7B优化
                quantization_config = GPTQConfig(
                    bits=4,  # INT4量化
                    group_size=128,  # 优化组大小
                    desc_act=False,  # 禁用描述激活（提升性能）
                    dtype=torch.float16,  # 🔧 修复：使用float16以兼容CUDA内核
                    # 添加Qwen7B专用参数
                    use_exllama=False,  # 禁用EXLLAMA（兼容性更好）
                    exllama_config={"version": 2}  # 使用EXLLAMA v2
                )

                # 加载GPTQ模型 - 优化参数设置
                model = AutoGPTQForCausalLM.from_quantized(
                    config["model_path"],
                    device="cuda:0",
                    use_safetensors=True,
                    use_triton=False,  # 保持False（除非需要Triton优化）
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.float16,  # 🔧 修复：使用float16以兼容CUDA内核
                    # 添加性能优化参数
                    max_memory={0: "24GB"},  # GPU内存限制
                    inject_fused_attention=False,  # 禁用融合注意力（我们的内核已优化）
                )

                # 关键优化：应用权重重排
                print("Applying weight rearrangement for Qwen7B...")
                model = rearrange_qwen_weights(model)
            else:
                raise ValueError(f"Unsupported quantization type: {config['quantization_type']}")
        else:
            # 非量化模型加载 (保持不变)
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )

        print(f"✅ Model loaded successfully with {config.get('quantization_type', 'no quantization')}")

        # 添加模型配置信息（用于适配器）
        if hasattr(model, 'config'):
            model.config.quantization_type = config.get("quantization_type", None)
            model.config.group_size = 128  # 确保组大小信息可用

    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
            print("⚠️ Fallback: Model loaded without quantization")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model even without quantization: {e2}")

    return model, tokenizer