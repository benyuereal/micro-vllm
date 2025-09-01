from core.engine import InferenceEngine
import time

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

if __name__ == "__main__":
    engine = InferenceEngine(MODEL_PATH)

    # 测试连续批处理
    prompts = [
        "写一段分片上传的代码",
        "请介绍北京这座城市：",
        "如何学习机器学习？",
        "Python编程语言有哪些主要优势？"
    ]

    print("Starting continuous batching...")
    start_time = time.time()

    results = engine.generate(
        prompts,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    gen_time = time.time() - start_time
    print(f"Generated in {gen_time:.2f} seconds")

    print("\nResults:")
    for i, prompt in enumerate(prompts):
        seq_id = id(prompt)
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Response: {results[seq_id]}")
        print("-" * 80)