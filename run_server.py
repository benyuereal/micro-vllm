from core.engine import InferenceEngine
import time

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

if __name__ == "__main__":
    print("Loading Qwen-7B model...")
    start_time = time.time()
    engine = InferenceEngine(MODEL_PATH)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    prompts = [
        "北京的旅游景点有",
        "如何学习机器学习",
        "Python编程有什么优势"
    ]

    print("Generating responses...")
    start_time = time.time()
    results = engine.generate(prompts, max_tokens=50)
    gen_time = time.time() - start_time

    print("\nResults:")
    for i, prompt in enumerate(prompts):
        seq_id = id(prompt)
        if seq_id in results:
            print(f"Prompt {i + 1}: {prompt}")
            print(f"Response: {results[seq_id]}")
            print("-" * 80)
        else:
            print(f"Prompt {i + 1}: {prompt} - No response generated")

    print(f"\nGenerated {len(prompts)} responses in {gen_time:.2f} seconds")