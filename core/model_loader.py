import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, \
    GPTQConfig  # GPTQConfig从transformers导入
import torch
from auto_gptq import AutoGPTQForCausalLM  # 仅导入AutoGPTQForCausalLM


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
                # 使用优化的GPTQ加载方式 - 移除GPTQConfig，使用auto_gptq的优化
                try:
                    model = AutoGPTQForCausalLM.from_quantized(
                        config["model_path"],
                        device="cuda:0",  # 指定GPU设备
                        use_safetensors=True,  # 使用safetensors格式
                        use_triton=True,  # 启用Triton优化（关键优化点）
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=torch.float16,  # 使用float16获得更好性能
                        # 优化参数
                        max_memory=None,  # 自动管理内存
                        device_map="auto",  # 自动分配设备
                        inject_fused_attention=True,  # 注入融合注意力
                        inject_fused_mlp=True,  # 注入融合MLP
                        use_cuda_fp16=True,  # 使用CUDA FP16
                        disable_exllama=False,  # 启用exllama优化
                        disable_exllamav2=True,  # 禁用exllamav2（如果exllama已启用）
                    )
                except Exception as gptq_error:
                    print(f"AutoGPTQ with optimizations failed: {gptq_error}")
                    print("Falling back to basic GPTQ configuration...")

                    # 回退到基本配置
                    model = AutoGPTQForCausalLM.from_quantized(
                        config["model_path"],
                        device="cuda:0",
                        use_safetensors=True,
                        use_triton=False,  # 如果Triton有问题则禁用
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=torch.float16
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

        # 启用优化设置
        model.eval()

        # CUDA优化设置
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

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