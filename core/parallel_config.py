# core/parallel_config.py (优化版，使用类封装但保持函数式接口)
import os  # 新增：用于读取环境变量
import torch
import torch.distributed as dist

class _ParallelContext:
    """内部类，用于封装状态"""
    world_group = None
    parallel_group = None
    is_main_rank = False

_ctx = _ParallelContext()

def setup():
    """
    初始化分布式环境（兼容单卡/多卡）
    - 多卡(torchrun)：正常初始化分布式
    - 单卡(python)：跳过初始化，使用默认值
    """
    # 新增：从环境变量读取分布式配置，单卡时默认值为0/1
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 新增：仅当多卡且未初始化时，才初始化分布式
    if world_size > 1 and not dist.is_initialized():
        # 自动选择后端（CUDA用nccl，CPU用gloo）
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
    
    # 原有逻辑：仅当分布式初始化后，才设置进程组
    if dist.is_initialized() and _ctx.world_group is None:
        _ctx.world_group = dist.group.WORLD
    
    # 新增：仅当分布式初始化后，才创建并行组
    if dist.is_initialized() and _ctx.parallel_group is None:
        _ctx.parallel_group = dist.new_group(ranks=list(range(get_world_size())))
    
    # 初始化 is_main_rank
    _ctx.is_main_rank = rank0()

def get_group():
    # 新增：分布式未初始化时返回None，避免后续调用报错
    return _ctx.parallel_group if dist.is_initialized() else None

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def rank0():
    """返回当前进程是否是主Rank（用于采样等操作）
    优先从环境变量读取，否则fallback到分布式初始化状态
    """
    # 优先从环境变量读取
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    
    # Fallback到分布式初始化状态
    return get_rank() == 0



def all_reduce(input_):
    if get_world_size() == 1:
        return input_
    # 新增：指定默认group为WORLD，避免parallel_group为None时报错
    dist.all_reduce(input_, group=get_group() or dist.group.WORLD)
    return input_

def barrier():
    if get_world_size() > 1 and dist.is_initialized():
        # 新增：指定默认group为WORLD，避免parallel_group为None时报错
        dist.barrier(group=get_group() or dist.group.WORLD)