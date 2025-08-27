import torch
import psutil
from typing import List, Dict


class MemoryManager:
    """内存管理器，负责监控和优化内存使用"""

    def __init__(self):
        self.gpu_memory_usage = 0
        self.cpu_memory_usage = 0

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "free": free,
                "total": total,
                "used": total - free
            }
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            # macOS Metal Performance Shaders
            return {
                "free": 0,  # MPS不提供内存信息
                "total": 0,
                "used": 0
            }
        return {"free": 0, "total": 0, "used": 0}

    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存信息"""
        memory = psutil.virtual_memory()
        return {
            "free": memory.available,
            "total": memory.total,
            "used": memory.used
        }

    def optimize_memory_usage(self, model: torch.nn.Module):
        """优化内存使用"""
        # 尝试使用更小的数据类型
        self._try_use_half_precision(model)

        # 尝试使用梯度检查点
        self._try_use_gradient_checkpointing(model)

    def _try_use_half_precision(self, model: torch.nn.Module):
        """尝试使用半精度"""
        if torch.cuda.is_available() or (hasattr(torch, 'mps') and torch.mps.is_available()):
            try:
                model.half()
                print("模型已转换为半精度")
            except Exception as e:
                print(f"无法转换为半精度: {e}")

    def _try_use_gradient_checkpointing(self, model: torch.nn.Module):
        """尝试使用梯度检查点"""
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print("已启用梯度检查点")
        except Exception as e:
            print(f"无法启用梯度检查点: {e}")