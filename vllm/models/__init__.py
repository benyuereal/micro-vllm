from .model_registry import register_model, get_model
from .qwen import QwenModel

# 自动注册所有支持的模型
register_model("Qwen")(QwenModel)
# 可以在这里添加更多模型注册，例如：
# register_model("LLaMA")(LLaMAModel)
# register_model("ChatGLM")(ChatGLMModel)

__all__ = ["register_model", "get_model", "QwenModel"]