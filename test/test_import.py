#!/usr/bin/env python3
"""
简单的导入测试
验证模块导入是否正常
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"🔍 项目根目录: {project_root}")
print(f"🔍 Python路径: {sys.path[:3]}")

try:
    from core.layer.gptq import GPTQCUDAFusion
    print("✅ GPTQCUDAFusion导入成功!")
except ImportError as e:
    print(f"❌ GPTQCUDAFusion导入失败: {e}")
    sys.exit(1)

try:
    from core.layer.optimized_qwen_layer import OptimizedQwenLayer
    print("✅ OptimizedQwenLayer导入成功!")
except ImportError as e:
    print(f"❌ OptimizedQwenLayer导入失败: {e}")

print("🎉 模块导入测试完成!")
