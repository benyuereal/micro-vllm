#!/usr/bin/env python3
"""
vLLM API服务器启动脚本
"""

import argparse
from vllm import APIServer


def main():
    parser = argparse.ArgumentParser(description="vLLM API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Maximum number of sequences")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")

    args = parser.parse_args()

    # 创建并启动服务器
    server = APIServer(host=args.host, port=args.port)
    server.initialize(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_seq_length=args.max_seq_length
    )

    print(f"Starting server on {args.host}:{args.port}")
    server.run()


if __name__ == "__main__":
    main()