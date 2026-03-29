"""
zip_matmul: 压缩权重 + 在线解压 matmul
========================================
思路：
  LLM 推理瓶颈是 HBM 带宽（权重从显存读取）。
  把权重以 TCA-TBE 格式常驻 GPU，每次 matmul 时先在片上解压，再做计算。
  节省 HBM 带宽 → 在小 batch（memory-bound）场景下有加速潜力。

接口：
  CompressedWeight      把 BF16 权重压缩并保存在 GPU 上
  zip_matmul(a, cw)     解压 + matmul_v3，结果与原始 matmul bit-close
"""

import torch
import time
import triton
import triton.language as tl
import time
from lossless import encode, decode, compressed_bytes

_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

# ---------------------------------------------------------------------------
# 压缩权重容器
# ---------------------------------------------------------------------------

class CompressedWeight:
    """将 BF16 权重压缩后常驻 GPU。"""

    def __init__(self, weight: torch.Tensor, top_k: int = 7):
        assert weight.is_cuda, "权重必须在 GPU 上"
        assert weight.dtype == torch.bfloat16
        self.shape = weight.shape
        # encode/decode 支持 CUDA tensor
        self.compressed = encode(weight, top_k=top_k)

    def decode(self) -> torch.Tensor:
        """解压，返回原始 BF16 权重（在 GPU 上）。"""
        return decode(self.compressed)

    @property
    def original_bytes(self) -> int:
        K, N = self.shape
        return K * N * 2  # BF16 = 2 bytes

    @property
    def compressed_size(self) -> int:
        return compressed_bytes(self.compressed)

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_size


# ---------------------------------------------------------------------------
# zip_matmul
# ---------------------------------------------------------------------------

def zip_matmul(a: torch.Tensor, cw: CompressedWeight,
               out: torch.Tensor = None) -> torch.Tensor:
    """
    解压 CompressedWeight 后调用 matmul_v3。

    Args:
        a  : 激活张量 (M, K)，BF16，在 GPU 上
        cw : 压缩权重，shape (K, N)
        out: 可选输出张量 (M, N)

    Returns:
        c  : (M, N) matmul 结果
    """
    weight = cw.decode()          # HBM 读压缩数据 + 解压（在 GPU 上完成）
    return matmul_v3(a, weight, out=out)




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 256, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_N': 32,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 5, 'num_warps': 4}),
    ],
    key=['N', 'K'],
)
@triton.jit
def matmul_kernel_v3(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr = tl.float16,
):
    # M=32 硬编码
    BLOCK_M: tl.constexpr = 32

    pid_n = tl.program_id(axis=0)

    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(OUTPUT_DTYPE))


def matmul_v3(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"
        assert out.device == a.device
        c = out

    output_dtype = _DTYPE_MAP.get(c.dtype, tl.float16)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    matmul_kernel_v3[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        OUTPUT_DTYPE=output_dtype,
    )
    return c

# ---------------------------------------------------------------------------
# 正确性验证
# ---------------------------------------------------------------------------

def verify_correctness():
    print("=== 正确性验证 ===\n")
    torch.manual_seed(0)
    device = "cuda"

    shapes = [
        (32, 4096,  4096,  "square [32,4096]x[4096,4096]"),
        (32, 11008, 4096,  "down proj [32,11008]x[11008,4096]"),
        (32, 4096,  12288, "QKV proj  [32,4096]x[4096,12288]"),
    ]

    all_pass = True
    for M, K, N, label in shapes:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(K, N, dtype=torch.bfloat16, device=device)

        cw = CompressedWeight(w)

        # 1. 权重解压 bit-wise 验证
        w_recon = cw.decode()
        bit_ok = torch.all(w.view(torch.int16) == w_recon.view(torch.int16)).item()

        # 2. matmul 正确性：用 matmul_v3(a, w) 做参考（同一套 Triton kernel）
        #    zip_matmul 内部 = decode(w) + matmul_v3，权重 bit-identical → 结果必须完全相同
        ref_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        out_buf = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        ref = matmul_v3(a, w,       out=ref_buf)
        out = zip_matmul(a, cw,     out=out_buf)

        matmul_ok = torch.all(ref.view(torch.int16) == out.view(torch.int16)).item()

        # 3. 与 torch.matmul 的数值误差（仅供参考，两套实现本就有浮点差异）
        ref_torch = torch.matmul(a, w)
        max_err = (ref_torch.float() - out.float()).abs().max().item()

        status = "PASS" if (bit_ok and matmul_ok) else "FAIL"
        if not (bit_ok and matmul_ok):
            all_pass = False

        print(f"[{label}]")
        print(f"  权重解压 bit-wise 无损  : {'YES' if bit_ok else 'NO'}")
        print(f"  zip_matmul vs matmul_v3: {'bit-identical' if matmul_ok else 'MISMATCH'}  [{status}]")
        print(f"  vs torch.matmul 最大误差: {max_err:.2e}  (cuBLAS vs Triton 正常浮点差异)")
        print(f"  压缩率: {cw.ratio:.2f}x  "
              f"({cw.original_bytes/1024:.0f} KB → {cw.compressed_size/1024:.0f} KB)")
        print()

    print("正确性验证:", "全部 PASS" if all_pass else "存在 FAIL")
    return all_pass


# ---------------------------------------------------------------------------
# 性能对比 Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    print("\n=== 性能对比 Benchmark ===\n")
    WARMUP  = 50
    REPEAT  = 500
    device  = "cuda"
    dtype   = torch.bfloat16

    # 与 matmul.py 保持一致的典型 LLM 形状（M=32 匹配 matmul_v3 的 BLOCK_M）
    shapes = [
        (32, 11008, 4096,  "down proj [32,11008]x[11008,4096]"),
        (32, 4096,  12288, "QKV proj  [32,4096]x[4096,12288]"),
        (32, 4096,  4096,  "FFN gate  [32,4096]x[4096,4096]"),
    ]

    hdr = f"{'Shape':<38} {'torch':>9} {'v3':>9} {'zip':>9} {'zip/v3':>8} {'压缩率':>7}"
    print(hdr)
    print("-" * len(hdr))

    for M, K, N, label in shapes:
        a = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(K, N, dtype=dtype, device=device)
        c = torch.empty(M, N, dtype=dtype, device=device)

        cw = CompressedWeight(w)

        # ---------- warmup ----------
        for _ in range(WARMUP):
            torch.matmul(a, w, out=c)
            matmul_v3(a, w, out=c)
            zip_matmul(a, cw, out=c)
        torch.cuda.synchronize()

        # ---------- torch.matmul ----------
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            torch.matmul(a, w, out=c)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / REPEAT * 1e6   # µs

        # ---------- matmul_v3（原始 BF16 权重）----------
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            matmul_v3(a, w, out=c)
        torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) / REPEAT * 1e6

        # ---------- zip_matmul（压缩权重 + 在线解压）----------
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            zip_matmul(a, cw, out=c)
        torch.cuda.synchronize()
        t_zip = (time.perf_counter() - t0) / REPEAT * 1e6

        ratio = cw.ratio

        print(f"{label:<38} {t_torch:>7.1f}µs  {t_v3:>7.1f}µs  {t_zip:>7.1f}µs  "
              f"{t_zip/t_v3:>7.2f}x  {ratio:>6.2f}x")

        # 各阶段耗时拆解（解压 vs 计算）
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = cw.decode()
        torch.cuda.synchronize()
        t_dec = (time.perf_counter() - t0) / REPEAT * 1e6

        print(f"  └─ 解压耗时: {t_dec:.1f}µs  计算耗时: {t_zip - t_dec:.1f}µs  "
              f"解压占比: {t_dec/t_zip*100:.1f}%")

    print()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_correctness()
    benchmark()
