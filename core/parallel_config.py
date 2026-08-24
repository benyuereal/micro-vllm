# core/parallel_config.py (优化版，使用类封装但保持函数式接口)
import os  # 新增：用于读取环境变量
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

class _ParallelContext:
    """内部类，用于封装状态"""
    world_group = None
    parallel_group = None
    is_main_rank = False
    # TP 性能：vLLM 编译版 custom allreduce（CUDA-IPC，单 kernel，无 NCCL 开销）。
    # NCCL 在 GPU2↔GPU5=SYS(跨 NUMA PCIe) 上 P2P 极慢（bs=32 56 次 allreduce=5.9ms/步），
    # custom AR 走 IPC 单 kernel（~0.05ms/次），是 vLLM TP=2 快 2.8x 的主因。
    custom_ar = None
    custom_ar_gloo = None

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
        # TP 修复：init_process_group 前必须先 set_device(rank)——NCCL communicator
        # 绑定到当前默认设备，若 rank1 在 cuda:0 上初始化，其 allreduce 会走错卡
        # （P2P 开启时直接 hang，P2P 关闭时退化为 host 中转慢 30%+）。
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
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

    # TP 性能：初始化 vLLM 编译版 custom allreduce（CUDA-IPC 单 kernel）。
    # 需一个非 NCCL（gloo）group 做 IPC handle 交换；NCCL group 仍用于 fallback。
    # 失败（无 vllm / 不支持）则 custom_ar=None，all_reduce 回退 NCCL。
    if dist.is_initialized() and get_world_size() > 1 and _ctx.custom_ar is None:
        try:
            if os.environ.get("MICRO_TP_NO_CUSTOM_AR") != "1":
                from vllm.distributed.device_communicators.custom_all_reduce import (
                    CustomAllreduce)
                _ctx.custom_ar_gloo = dist.new_group(
                    ranks=list(range(get_world_size())), backend="gloo")
                _ctx.custom_ar = CustomAllreduce(
                    group=_ctx.custom_ar_gloo, device=f"cuda:{rank}")
                if _ctx.custom_ar.disabled:
                    logger.info("custom allreduce disabled → 回退 NCCL")
                    _ctx.custom_ar = None
                else:
                    logger.info("✅ vLLM custom allreduce 启用（CUDA-IPC 单 kernel）")
        except Exception as e:
            logger.warning(f"custom allreduce 初始化失败，回退 NCCL: {e}")
            _ctx.custom_ar = None

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



def all_reduce(input_, out=None):
    """TP allreduce。优先用 vLLM 编译版 custom allreduce（CUDA-IPC 单 kernel，
    ~0.03ms/次 in-graph），回退 NCCL（SYS 拓扑 P2P 慢，~0.1ms/次）。

    custom AR 是 out-of-place：结果写 out（须为【常驻】buffer，graph 捕获时
    register_graph_buffers 会 IPC 注册 input/out 指针，replay 时 kernel 直读直写）。
    out=None 时临时分配（仅 eager 路径用，如 prefill）。
    """
    if get_world_size() == 1:
        return input_
    # 诊断开关：MICRO_TP_NO_ALLREDUCE=1 跳过 allreduce（仅用于量化通信开销，
    # 输出不正确，勿用于正确性验证）。
    if os.environ.get("MICRO_TP_NO_ALLREDUCE") == "1":
        return input_
    ca = _ctx.custom_ar
    if ca is not None and not ca.disabled and ca.should_custom_ar(input_):
        _ctx._ar_custom = getattr(_ctx, "_ar_custom", 0) + 1
        if out is None:
            out = torch.empty_like(input_)
        # graph 捕获中：registered=True（kernel 直读 input，graph 安全，捕获后
        # register_graph_buffers 会 IPC 注册 input/out 指针）；eager/warmup：
        # registered=False（memcpy 进预注册 buffer，可加载 cubin，非 graph 安全）。
        registered = torch.cuda.is_current_stream_capturing()
        return ca.all_reduce(input_, out=out, registered=registered)
    # 回退 NCCL（in-place）
    _ctx._ar_nccl = getattr(_ctx, "_ar_nccl", 0) + 1
    dist.all_reduce(input_, group=get_group() or dist.group.WORLD)
    return input_


def register_graph_buffers():
    """graph 捕获后调用：交换各 rank 的 graph buffer IPC handle，使 replay 时
    custom AR kernel 能经 IPC 读 peer 的 allreduce input。所有 rank 须同序调用。"""
    ca = _ctx.custom_ar
    if ca is not None and not ca.disabled:
        ca.register_graph_buffers()

def barrier():
    if get_world_size() > 1 and dist.is_initialized():
        # 新增：指定默认group为WORLD，避免parallel_group为None时报错
        dist.barrier(group=get_group() or dist.group.WORLD)