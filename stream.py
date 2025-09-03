# test_stream.py
from core.engine import InferenceEngine
import time
import threading
from queue import Queue

MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

class StreamTester:
    def __init__(self, model_path=MODEL_PATH):
        print("Loading model for stream testing...")
        self.engine = InferenceEngine(model_path)
        print(f"Model loaded on device: {self.engine.device}")

    def test_single_stream(self, prompt, max_tokens=1024):
        """测试单个流式输出"""
        print(f"\n{'=' * 60}")
        print(f"测试单个流式输出")
        print(f"输入: {prompt}")
        print(f"{'=' * 60}")

        full_text = ""
        token_count = 0

        # 使用流式生成器
        for token, text in self.engine.stream_generate(prompt, max_tokens=max_tokens):
            print(text, end="", flush=True)
            full_text += text
            token_count += 1
            time.sleep(0.05)  # 模拟实时输出效果

        print(f"\n\n生成完成，共 {token_count} 个token")
        return full_text

    def test_stream_with_callback(self, prompt, max_tokens=30):
        """使用回调函数测试流式输出"""
        print(f"\n{'=' * 60}")
        print(f"测试回调函数流式输出")
        print(f"输入: {prompt}")
        print(f"{'=' * 60}")

        # 创建队列用于接收回调
        token_queue = Queue()

        # 添加请求并获取序列ID
        seq_id = self.engine.add_request(prompt, max_tokens=max_tokens)

        # 定义回调函数
        def callback(token, text):
            token_queue.put((token, text))

        # 注册回调
        self.engine.register_stream_callback(seq_id, callback)

        try:
            full_text = ""
            token_count = 0

            # 手动执行step直到完成
            while token_count < max_tokens:
                self.engine.step()

                # 检查是否有新token
                while not token_queue.empty():
                    token, text = token_queue.get()
                    print(text, end="", flush=True)
                    full_text += text
                    token_count += 1

                    # 如果遇到结束标记，停止生成
                    if token == self.engine.eos_token_id:
                        print(f"\n\n遇到结束标记，生成完成，共 {token_count} 个token")
                        return full_text

                # 短暂休眠
                time.sleep(0.01)

            print(f"\n\n达到最大token限制，生成完成，共 {token_count} 个token")
            return full_text
        finally:
            # 清理回调
            self.engine.unregister_stream_callback(seq_id)

    def test_multiple_streams(self, prompts, max_tokens=30):
        """测试多个并发的流式输出"""
        print(f"\n{'=' * 60}")
        print(f"测试多个并发流式输出")
        print(f"输入数量: {len(prompts)}")
        print(f"{'=' * 60}")

        results = {}

        # 为每个提示创建一个线程
        threads = []
        for i, prompt in enumerate(prompts):
            thread = threading.Thread(
                target=self._run_single_stream_thread,
                args=(i, prompt, max_tokens, results)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        return results

    def _run_single_stream_thread(self, idx, prompt, max_tokens, results):
        """在单独线程中运行单个流式输出"""
        try:
            result = self.test_single_stream(f"[线程{idx}] {prompt}", max_tokens)
            results[idx] = result
        except Exception as e:
            results[idx] = f"错误: {str(e)}"

    def test_interactive_stream(self):
        """交互式流式输出测试"""
        print(f"\n{'=' * 60}")
        print(f"交互式流式输出测试")
        print("输入 'quit' 退出")
        print(f"{'=' * 60}")

        while True:
            prompt = input("\n请输入提示语: ")
            if prompt.lower() == 'quit':
                break

            max_tokens = input("请输入最大token数 (默认50): ") or "50"
            try:
                max_tokens = int(max_tokens)
            except ValueError:
                max_tokens = 50

            print("生成结果:")
            self.test_single_stream(prompt, max_tokens)


if __name__ == "__main__":
    tester = StreamTester()

    # 测试单个流式输出
    prompt = "帮我写一段文件分片上传的代码"
    result1 = tester.test_single_stream(prompt)
    #
    # # 测试回调函数方式
    # prompt2 = "深度学习的主要应用包括"
    # result2 = tester.test_stream_with_callback(prompt2)

    # 测试多个并发流
    # prompts = [
    #     "春天的花开",
    #     "夏天的风",
    #     "秋天的月亮",
    #     "冬天的雪"
    # ]
    # results = tester.test_multiple_streams(prompts, max_tokens=100)
    #
    # # 交互式测试
    # tester.test_interactive_stream()

