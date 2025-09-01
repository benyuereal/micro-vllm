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
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        traceback.print_exc()
        exit(1)

    # 测试不同长度的提示
    prompts = [
        "北京",  # 短提示
        "如何学习机器学习？",  # 中等提示
        "Python编程语言有哪些主要优势？请详细说明。"  # 长提示
    ]

    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()

    try:
        results = engine.generate(prompts, max_tokens=50)
        gen_time = time.time() - start_time
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        traceback.print_exc()
        exit(1)

    print("\nResults:")
    for i, prompt in enumerate(prompts):
        seq_id = id(prompt)
        if seq_id in results:
            print(f"Prompt {i + 1}: {prompt}")
            print(f"Response: {results[seq_id]}")
            print(f"Length: {len(results[seq_id])} characters")
            print("-" * 80)
        else:
            print(f"Prompt {i + 1}: {prompt} - No response generated")

    tokens_generated = sum(len(response) for response in results.values())
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

    print(f"\nGenerated {tokens_generated} tokens in {gen_time:.2f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print("Done!")