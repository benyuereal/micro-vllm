import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .cache import KVCache
from .scheduler import Scheduler


class InferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.kv_cache = KVCache()
        self.scheduler = Scheduler()

    def prefill(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=True
        )
        return outputs

    def decode(self, token, past_key_values):
        inputs = torch.tensor([[token]], device=self.device)
        outputs = self.model(
            input_ids=inputs,
            past_key_values=past_key_values,
            use_cache=True
        )
        return outputs.logits, outputs.past_key_values

    def process_request(self, request):
        # 实际调度处理逻辑
        pass