# run_server.py
from core.engine import InferenceEngine

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

if __name__ == "__main__":
    print("Loading Qwen-7B model...")
    engine = InferenceEngine(MODEL_PATH)

    prompts = [
        "北京的旅游景点有",
        "如何学习机器学习",
        "Python编程有什么优势"
    ]

    print("Generating responses...")
    results = engine.generate(prompts, max_tokens=5000)

    print("\nResults:")
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Response: {results[id(prompt)]}")
        print("-" * 80)