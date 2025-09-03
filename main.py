from core.engine import InferenceEngine
import time
import traceback

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

if __name__ == "__main__":
    print("Loading Qwen-7B model...")
    start_time = time.time()
    try:
        engine = InferenceEngine(MODEL_PATH)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # 检查模型状态
        print(f"Model device: {engine.device}")
        print(f"Model dtype: {next(engine.model.parameters()).dtype}")

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        traceback.print_exc()
        exit(1)

    # 测试不同长度的提示
    prompts = [
        "写一段分片上传的代码",
        "请介绍北京这座城市：",  # 短提示
        "如何学习机器学习？",  # 中等提示
        "Python编程语言有哪些主要优势？请详细说明。"  # 长提示
    ]

    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()

    try:
        # 使用更高质量的生成参数
        results = engine.generate(
            prompts,
            max_steps=1024
        )
        gen_time = time.time() - start_time
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        traceback.print_exc()
        exit(1)

    print("\nResults:")
    # ✅ 方法 1：直接遍历 key 和 value
    for seq,  text in results.items():
        print(f"seq_id: {seq.seq_id}, prompt: {seq.prompt}")
        seq_id = seq.seq_id
        print(f"Prompt : {seq.prompt}")
        print(f"Response: {text}")
        print(f"Length: {len(text)} characters")
        print("-" * 80)


    tokens_generated = sum(len(response) for response in results.values())
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

    print(f"\nGenerated {tokens_generated} tokens in {gen_time:.2f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print("Done!")