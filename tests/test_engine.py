import unittest
from core.engine import InferenceEngine
import time


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()
        cls.engine = InferenceEngine("/data/model/qwen/Qwen-7B-Chat")
        cls.load_time = time.time() - cls.start_time
        print(f"\nModel loaded in {cls.load_time:.2f} seconds")

    def test_single_generation(self):
        start_time = time.time()
        results = self.engine.generate(["Hello, how are you?"], max_tokens=20)
        gen_time = time.time() - start_time

        self.assertEqual(len(results), 1)
        response = list(results.values())[0]
        self.assertGreater(len(response), 5)
        self.assertIn("Hello", response)
        print(f"Single generation completed in {gen_time:.2f} seconds")

    def test_batch_generation(self):
        prompts = ["What is AI?", "Explain quantum computing"]
        start_time = time.time()
        results = self.engine.generate(prompts, max_tokens=30)
        gen_time = time.time() - start_time

        self.assertEqual(len(results), 2)
        for prompt in prompts:
            response = results[id(prompt)]
            self.assertTrue(response.startswith(prompt.split()[0]))
        print(f"Batch generation ({len(prompts)} prompts) completed in {gen_time:.2f} seconds")

    def test_chinese_generation(self):
        start_time = time.time()
        results = self.engine.generate(["人工智能是什么"], max_tokens=40)
        gen_time = time.time() - start_time

        response = list(results.values())[0]
        self.assertIn("人工智能", response)
        print(f"Chinese generation completed in {gen_time:.2f} seconds")


if __name__ == "__main__":
    unittest.main()