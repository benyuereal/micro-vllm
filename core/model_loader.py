from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import argparse
import torch

# 本地模型搜索根目录（按顺序查找）。可通过环境变量 MICRO_VLLM_MODEL_DIRS 覆盖
# （冒号分隔，类似 PATH）。短名会在这里递归找含 config.json 的目录。
_DEFAULT_MODEL_DIRS = ["/models"]
MODEL_NAME_ALIASES = {
    # 短名 / 常见别名 → 规范名（用于短名模糊匹配时去歧义）
    "deepseek": "DeepSeek-V2-Lite",
    "deepseek-v2-lite": "DeepSeek-V2-Lite",
    "qwen": "Qwen-7B-Chat",
    "qwen-7b": "Qwen-7B-Chat",
    "qwen-7b-chat": "Qwen-7B-Chat",
    "qwen3": "Qwen3-0.6B",
    "qwen3-0.6b": "Qwen3-0.6B",
}


def _model_dirs():
    env = os.environ.get("MICRO_VLLM_MODEL_DIRS")
    if env:
        return [d for d in env.split(os.pathsep) if d]
    return _DEFAULT_MODEL_DIRS


def _is_model_root(path: str) -> bool:
    """判断 path 是否是一个可加载的模型根目录（含 config.json）。"""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def resolve_model_path(name: str) -> str:
    """把 --model 参数解析成可加载的模型根目录路径。

    支持两种输入：
      1. 路径（绝对/相对，含 config.json）——直接返回规范化的绝对路径。
      2. 短名/模型名（如 Qwen-7B-Chat、deepseek）——在本地模型搜索根目录下
         递归查找名字匹配且含 config.json 的目录。

    解析顺序：别名 → 直接路径 → 各搜索根下的 {name} → 递归模糊匹配（名字包含 name
    且含 config.json）。找不到则抛 FileNotFoundError 并列出候选。
    """
    if not name:
        raise ValueError("model name/path is empty")

    # 1) 别名
    alias = MODEL_NAME_ALIASES.get(name.lower())
    if alias is not None:
        name = alias

    # 2) 直接是路径
    if _is_model_root(name):
        return os.path.abspath(name)

    # 3) 各搜索根下的 {name}（直接拼接）
    for root in _model_dirs():
        cand = os.path.join(root, name)
        if _is_model_root(cand):
            return os.path.abspath(cand)

    # 4) 递归模糊匹配：在搜索根下找名字含 name 且含 config.json 的目录
    needle = name.lower()
    candidates = []
    for root in _model_dirs():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过常见的非模型目录，加速遍历
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
            if "config.json" not in filenames:
                continue
            base = os.path.basename(dirpath)
            if needle in base.lower():
                candidates.append(dirpath)

    if len(candidates) == 1:
        return os.path.abspath(candidates[0])
    if len(candidates) > 1:
        # 多个候选：优先选名字最短（最贴近）的；并列则报错列出
        candidates.sort(key=lambda p: (len(os.path.basename(p)), p))
        best = candidates[0]
        if len(os.path.basename(best)) == len(os.path.basename(candidates[1])):
            raise FileNotFoundError(
                f"短名 {name!r} 匹配到多个模型根，请用完整路径：\n  " +
                "\n  ".join(candidates))
        return os.path.abspath(best)

    raise FileNotFoundError(
        f"找不到模型 {name!r}。已查找路径：{name}（作为路径）"
        f"及搜索根 {_model_dirs()} 下的递归匹配。"
        f"可用环境变量 MICRO_VLLM_MODEL_DIRS 追加搜索根（冒号分隔）。")


def parse_model_args(argv=None) -> argparse.Namespace:
    """解析模型加载相关的 CLI 参数（vLLM 风格）。

    --model       模型路径或短名（必填，或用 --model-name / 环境变量 MODEL_NAME）
    --model-name  同 --model 的别名
    其余参数（max-batch-size / port 等）由调用方按需追加，这里只管模型定位。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", dest="model", default=None,
                        help="模型路径或短名（如 /models/Qwen-7B-Chat 或 Qwen-7B-Chat）")
    parser.add_argument("--model-name", dest="model_name", default=None,
                        help="--model 的别名")
    args, _ = parser.parse_known_args(argv)
    return args


def get_model_path_from_cli(argv=None, default: str = None) -> str:
    """统一的模型路径解析入口（CLI 参数 > 环境变量 > default）。

    优先级：--model / --model-name > 环境变量 MODEL_NAME > default。
    解析到的短名会经 resolve_model_path 展开成本地完整路径。
    """
    args = parse_model_args(argv)
    name = args.model or args.model_name or os.environ.get("MODEL_NAME") or default
    if name is None:
        raise ValueError(
            "未指定模型。请用 --model <路径或短名>，或设置环境变量 MODEL_NAME。")
    return resolve_model_path(name)


# 必要修改 1：新增 device 参数（默认 None，兼容原有调用）
def load_model(model_path, device=None):
    try:
        # 你原有 tokenizer 逻辑完全不动
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True
        )
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")  # 应该输出 "QWenTokenizer"
        print(f"[DEBUG] eos_token_id: {tokenizer.eos_token_id}")  # Qwen-7B-Chat 通常是 151643


    except Exception as e:
        print(f"Fast tokenizer failed: {e}, trying slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

    # 核心逻辑开始
    if device is not None:
        # TP 场景：禁用 accelerate 自动分片，强制到指定设备
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,  # 关键：指定具体设备（如 cuda:1）
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        # 兼容你原有逻辑：没传 device 时仍用 auto
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
    return model, tokenizer