import unittest


from vllm.configs import qwen_7b as config
from ..models.qwen  import QwenEngine

class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = QwenEngine(config.MODEL_PATH)

    def test_prefill(self):
        prompt = "你好，介绍一下你自己"
        output = self.engine.prefill(prompt)
        self.assertIsNotNone(output)
        print("测试输出:", output)

    def test_prompt_format(self):
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
            {"role": "user", "content": "介绍一下量子计算"}
        ]
        prompt = self.engine.build_prompt(messages)
        self.assertTrue("<|im_start|>" in prompt)


if __name__ == "__main__":
    unittest.main()