# vllm/response.py
from dataclasses import asdict, is_dataclass
import json
from typing import Any, Dict, List, Optional, Union

import torch

from .schema import Response

class ResponseEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持数据类和Tensor"""
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() > 1 else obj.item()
        return super().default(obj)

def create_success_response(request_id: int, generated_text: str) -> Response:
    """创建成功响应"""
    return Response(
        request_id=request_id,
        generated_text=generated_text,
        success=True
    )

def create_error_response(request_id: int, error_message: str) -> Response:
    """创建错误响应"""
    return Response(
        request_id=request_id,
        generated_text="",
        success=False,
        error_message=error_message
    )

def create_incremental_response(request_id: int, incremental_text: str) -> Response:
    """创建增量生成响应"""
    return Response(
        request_id=request_id,
        generated_text=incremental_text,
        success=False
    )

def format_response(response: Response, as_json: bool = True) -> Union[Dict, Response]:
    """格式化响应为JSON或原生对象"""
    if as_json:
        return json.dumps(asdict(response), cls=ResponseEncoder)
    return response

def batch_format_responses(responses: List[Response], as_json: bool = True) -> List[Any]:
    """批量格式化响应"""
    return [format_response(r, as_json) for r in responses]