import torch
from ..core.engine import InferenceEngine


class QwenEngine(InferenceEngine):
    def __init__(self, model_path):
        super().__init__(model_path)

    def apply_optimizations(self):
        # 应用模型特定优化
        self.model = torch.compile(self.model)

    def build_prompt(self, messages):
        # 构建Qwen专用prompt格式
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt