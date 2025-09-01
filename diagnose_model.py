# diagnose_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"


def diagnose_model():
    print("Diagnosing Qwen model...")

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # 测试分词器
    test_text = "北京"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Tokenizer test: '{test_text}' -> {tokens} -> '{decoded}'")

    # 简单生成测试
    input_ids = tokenizer.encode(test_text, return_tensors="pt").cuda()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Direct generation test: '{generated_text}'")

    return model, tokenizer


if __name__ == "__main__":
    diagnose_model()