from typing import Dict, Type

_MODEL_REGISTRY = {}

def register_model(name: str):
    """注册模型类的装饰器"""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str):
    """根据名称获取模型类"""
    if name not in _MODEL_REGISTRY:
        # 尝试从名称中提取模型类型
        model_type = name.split("/")[-1].split("-")[0]
        if model_type in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[model_type]
        raise ValueError(f"Model {name} not found in registry")
    return _MODEL_REGISTRY[name]