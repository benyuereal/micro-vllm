import torch
import time
from config import Config
from model_loader import load_model_and_tokenizer, prepare_model_for_inference
from cache_manager import KVCacheManager
from scheduler import Scheduler
from engine import InferenceEngine
from request import Request


def print_colored(text, color):
    """带颜色打印输出"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")


class ModelTester:
    """模型测试器"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kv_cache_manager = None
        self.scheduler = None
        self.engine = None
        self.setup_completed = False

    def setup(self):
        """设置测试环境"""
        print_colored("Setting up test environment...", "blue")
        start_time = time.time()

        # 加载模型和tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.model = prepare_model_for_inference(self.model)

        # 获取模型配置
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads

        # 初始化KV缓存管理器
        self.kv_cache_manager = KVCacheManager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim
        )

        # 初始化调度器
        self.scheduler = Scheduler(self.kv_cache_manager, max_batch_size=Config.MAX_BATCH_SIZE)

        # 初始化推理引擎
        self.engine = InferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            kv_cache_manager=self.kv_cache_manager,
            scheduler=self.scheduler
        )
        self.engine.start()

        setup_time = time.time() - start_time
        print_colored(f"Setup completed in {setup_time:.2f} seconds", "green")
        self.setup_completed = True

    def direct_inference_test(self, prompt, max_tokens=10):
        """修复后的直接推理测试"""
        try:
            # 正确的输入处理
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,  # 确保填充
                return_attention_mask=True
            )

            # 移动到GPU
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

            # 调试打印
            print(f"Input IDs shape: {inputs['input_ids'].shape}")
            print(f"Attention Mask shape: {inputs['attention_mask'].shape}")

            with torch.no_grad():
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Input device: {inputs['input_ids'].device}")
                print(f"Input example: {inputs['input_ids'][0][:5]}")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id  # 必须设置
                )

            # 检查输出
            if outputs is None or outputs.numel() == 0:
                raise RuntimeError("模型返回空输出")

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print_colored(f"Response: {response}", "green")
            return response

        except Exception as e:
            print_colored(f"推理错误: {e}", "red")
            import traceback
            traceback.print_exc()

    def vllm_inference_test(self, prompt, max_tokens=10, temperature=0.7, top_p=0.9):
        """使用vLLM框架进行推理测试"""
        if not self.setup_completed:
            self.setup()

        print_colored("\n[vLLM Framework Test]", "yellow")
        print_colored(f"Prompt: {prompt}", "blue")
        print_colored(f"Params: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}", "blue")

        # 创建请求
        request = Request.from_prompt(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # 添加到调度器
        success = self.scheduler.add_request(request)
        if not success:
            print_colored("Failed to add request to scheduler", "red")
            return None

        # 等待请求完成
        start_time = time.time()
        while not request.finished:
            time.sleep(0.01)

        # 获取响应
        response = self.tokenizer.decode(request.output_tokens, skip_special_tokens=True)
        inference_time = time.time() - start_time

        print_colored(f"Response: {response}", "green")
        print_colored(f"Time: {inference_time:.4f}s | Tokens: {len(request.output_tokens)}", "blue")
        return response

    def batch_inference_test(self, prompts, max_tokens=10):
        """批量推理测试"""
        if not self.setup_completed:
            self.setup()

        print_colored("\n[Batch Inference Test]", "yellow")
        requests = []
        start_time = time.time()

        # 添加所有请求
        for i, prompt in enumerate(prompts):
            request = Request.from_prompt(
                prompt,
                max_tokens=max_tokens,
                request_id=f"batch_{i}"
            )
            self.scheduler.add_request(request)
            requests.append(request)
            print_colored(f"Added request {i}: {prompt[:30]}...", "blue")

        # 等待所有请求完成
        while any(not req.finished for req in requests):
            time.sleep(0.1)

        # 收集结果
        responses = []
        for i, request in enumerate(requests):
            response = self.tokenizer.decode(request.output_tokens, skip_special_tokens=True)
            responses.append(response)
            print_colored(f"\nRequest {i} response:", "yellow")
            print_colored(response, "green")

        total_time = time.time() - start_time
        total_tokens = sum(len(req.output_tokens) for req in requests)
        print_colored(f"\nTotal time: {total_time:.4f}s | Total tokens: {total_tokens}", "blue")

        return responses

    def test_model_output(self):
        """测试模型输出功能"""
        test_prompts = [
            "中国的首都是",
            "请用Python写一个快速排序算法",
            "解释量子计算机的工作原理",
            "将下列英文翻译成中文: 'Large language models have revolutionized natural language processing.'"
        ]

        # 测试直接推理
        print_colored("=" * 50, "yellow")
        print_colored("Starting Direct Inference Tests", "blue")
        print_colored("=" * 50, "yellow")

        for prompt in test_prompts[:2]:
            self.direct_inference_test(prompt)

        # 测试vLLM框架推理
        print_colored("\n" + "=" * 50, "yellow")
        print_colored("Starting vLLM Framework Tests", "blue")
        print_colored("=" * 50, "yellow")

        for i, prompt in enumerate(test_prompts):
            response = self.vllm_inference_test(prompt, max_tokens=15 + i * 5)

        # 测试批量推理
        print_colored("\n" + "=" * 50, "yellow")
        print_colored("Starting Batch Inference Test", "blue")
        print_colored("=" * 50, "yellow")

        self.batch_inference_test(test_prompts, max_tokens=15)

        print_colored("\nAll tests completed!", "green")

    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine.stop()
        print_colored("Test environment cleaned up", "green")


if __name__ == "__main__":
    tester = ModelTester()
    try:
        tester.test_model_output()
    except Exception as e:
        print_colored(f"Error during testing: {e}", "red")
    finally:
        tester.cleanup()