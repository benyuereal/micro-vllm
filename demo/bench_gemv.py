"""L2-flushed GEMV 基准：测真实 HBM 效率（隔离基准会被 L2 污染）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
from kernel.gemv import gemv_v2

# L2 flush buffer（L20 L2 = 96MB，用 200MB 确保冲刷）
_flush = torch.empty(200 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')

def flush_l2():
    _flush.fill_(0.0)

def bench(fn, iters=30):
    # 每次迭代前 flush L2，测真实 HBM 读
    times = []
    for _ in range(iters):
        flush_l2()
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    times.sort()
    return times[len(times)//2] * 1e6  # 中位数 us

sizes = [('GDN qkv', 6144, 1024), ('GDN z', 2048, 1024), ('GDN o', 1024, 2048),
         ('full qkv', 5120, 1024), ('full o', 1024, 2048),
         ('MLP gu', 7168, 1024), ('MLP d', 1024, 3584), ('lm_head', 248320, 1024)]
print('%-10s %8s %8s %10s %8s' % ('name', 'N', 'K', 'us', 'GB/s'))
for name, N, K in sizes:
    x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(1, N, dtype=torch.bfloat16, device='cuda')
    t = bench(lambda: gemv_v2(x, w, out))
    bw = N * K * 2 / (t * 1e-6) / 1e9
    print('%-10s %8d %8d %10.1f %8.0f' % (name, N, K, t, bw))
