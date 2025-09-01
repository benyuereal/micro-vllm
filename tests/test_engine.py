# tests/test_engine.py
import unittest
from core.engine import InferenceEngine


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InferenceEngine("/data/model/qwen/Qwen-7B-Chat")

    def test_single_generation(self):
        result = self.engine.generate(["Hello, how are you?"], max_tokens=10)
        self.assertEqual(len(result), 1)
        response = list(result.values())[0]
        self.assertGreater(len(response), 5)
        self.assertIn("Hello", response)

    def test_batch_generation(self):
        prompts = ["What is AI?", "Explain quantum computing"]
        results = self.engine.generate(prompts, max_tokens=15)
        self.assertEqual(len(results), 2)
        for prompt, response in zip(prompts, results.values()):
            self.assertTrue(response.startswith(prompt.split()[0]))

    def test_chinese_generation(self):
        result = self.engine.generate(["人工智能是什么"], max_tokens=20)
        response = list(result.values())[0]
        self.assertIn("人工智能", response)


if __name__ == "__main__":
    unittest.main()