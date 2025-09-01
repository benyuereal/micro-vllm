# main.py (更新)
from core.engine import InferenceEngine
import time
import traceback

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

if __name__ == "__main__":
    print("Loading Qwen-7B model with continuous batching...")
    start_time = time.time()
    try:
        engine = InferenceEngine(MODEL_PATH, max_batch_size=4)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Model device: {engine.device}")
        print(f"Model dtype: {next(engine.model.parameters()).dtype}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        traceback.print_exc()
        exit(1)

    # 测试不同长度的提示
    prompts = [
        "写一段分片上传的代码",
        "请介绍北京这座城市：",
        "如何学习机器学习？",
        "Python编程语言有哪些主要优势？请详细说明。",
        "解释一下量子计算的基本原理",
        "写一首关于春天的诗",
        "推荐几本值得阅读的科幻小说",
        "如何提高深度学习模型的泛化能力？"
    ]

    print(f"Generating responses for {len(prompts)} prompts with continuous batching...")
    start_time = time.time()

    try:
        results = engine.generate(
            prompts,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
        gen_time = time.time() - start_time
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        traceback.print_exc()
        exit(1)

    print("\nResults:")
    for i, prompt in enumerate(prompts):
        response = results.get(i, "No response generated")
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Response: {response}")
        print(f"Length: {len(response)} characters")
        print("-" * 80)

    tokens_generated = sum(len(response) for response in results.values())
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

    print(f"\nGenerated {tokens_generated} tokens in {gen_time:.2f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print("Continuous batching test completed!")