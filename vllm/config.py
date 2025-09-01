import torch


class Config:
    # 模型路径
    MODEL_PATH = "/data/model/qwen/Qwen-7B-Chat"

    # 推理配置
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # KV缓存配置
    BLOCK_SIZE = 16
    MAX_BLOCKS = 1024
    GPU_MEMORY_UTILIZATION = 0.9

    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8000