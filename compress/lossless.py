import torch


# -----------------------------------------------------------------------------
# TCA-TBE (Tensor-Core-Aware Triple Bitmap Encoding)
# 针对 BF16 LLM 权重的无损压缩算法
#
# BF16 位布局: [S(1)] [Exponent(8)] [Mantissa(7)]
#
# 核心思路：
#   LLM 权重的指数分布高度集中，Top-K 高频指数覆盖绝大多数元素。
#   对"主流"元素只需存 3-bit 指数编码 + 8-bit Sign/Mantissa = 11 bit（<原始 16 bit）。
#   剩余离群元素以原始 BF16 完整存储。
#
# 编码流程：
#   1. 拆解 BF16 → (Sign, Exponent, Mantissa)
#   2. 统计指数频率，选 Top-K（K ≤ 7）高频指数
#   3. 映射：指数 → 3-bit code，0~(K-1) 为主流，7 为离群
#   4. 将 code 的 3 个 bit 平面分别向量化打包成位图 B0/B1/B2
#   5. 非离群元素：存 packed_sm = Sign(1) | Mantissa(7)（共 8 bit）
#   6. 离群元素：存完整 BF16 值 + 其在展平数组中的下标
# -----------------------------------------------------------------------------

OUTLIER_CODE = 7   # 3-bit 编码中 7 保留给离群值，最多支持 7 个主流指数


# ---------------------------------------------------------------------------
# 位操作工具（向量化，无 Python 循环）
# ---------------------------------------------------------------------------

def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """将 bool 张量向量化打包成 uint8（LSB first）。"""
    n = bits.numel()
    pad = (8 - n % 8) % 8
    if pad:
        bits = torch.cat([bits, bits.new_zeros(pad)])
    # (n_bytes, 8) × 2^shift → sum 得到每字节值
    b = bits.view(-1, 8).to(torch.uint8)
    shift = torch.arange(8, dtype=torch.uint8, device=bits.device)
    return (b * (1 << shift)).sum(dim=1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    """从 uint8 向量化还原 bool 张量，取前 n 位。"""
    shift = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(1) >> shift) & 1).bool().reshape(-1)
    return bits[:n]


# ---------------------------------------------------------------------------
# BF16 分量拆解 / 组合（用 int16→int32 避开 CPU 上 uint16 运算限制）
# ---------------------------------------------------------------------------

def bf16_to_components(x: torch.Tensor):
    """BF16 1-D → (sign: int32, exp: int32, mant: int32)。"""
    u = x.flatten().view(torch.int16).to(torch.int32)
    sign = (u >> 15) & 0x1
    exp  = (u >>  7) & 0xFF
    mant =  u        & 0x7F
    return sign, exp, mant


def components_to_bf16(sign: torch.Tensor,
                       exp:  torch.Tensor,
                       mant: torch.Tensor) -> torch.Tensor:
    """(sign, exp, mant) → BF16 1-D 张量。"""
    u = (sign.to(torch.int32) << 15) | \
        (exp .to(torch.int32) <<  7) | \
        (mant.to(torch.int32) & 0x7F)
    return u.to(torch.int16).view(torch.bfloat16)


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------

def encode(weight: torch.Tensor, top_k: int = OUTLIER_CODE) -> dict:
    """
    对 BF16 权重张量进行 TCA-TBE 无损编码。

    压缩字典中需持久化的字段：
        shape           : 原始 shape
        num_elem        : 元素总数
        top_exps        : Top-K 指数值（int32，长度 ≤ top_k）
        packed_b0/b1/b2 : code 的三个 bit 平面，各打包为 uint8
        packed_sm       : 非离群元素的 packed Sign|Mantissa（uint8）
        outlier_idx     : 离群元素的下标（int32）
        outlier_vals    : 离群元素的完整 BF16 值
    """
    assert weight.dtype == torch.bfloat16, "仅支持 BF16 权重"
    assert 1 <= top_k <= OUTLIER_CODE, f"top_k 需在 [1, {OUTLIER_CODE}] 之间"

    sign, exp, mant = bf16_to_components(weight.flatten())
    num_elem = sign.numel()

    # ---- 统计高频指数，建立 exp → code 映射表（256 项，默认 OUTLIER_CODE）----
    exp_counts = torch.bincount(exp.to(torch.int64), minlength=256)
    top_exps = torch.argsort(exp_counts, descending=True)[:top_k]  # int64

    device = weight.device
    exp_map = torch.full((256,), OUTLIER_CODE, dtype=torch.uint8, device=device)
    for i, e in enumerate(top_exps.tolist()):
        exp_map[e] = i

    code = exp_map[exp]   # uint8，每个元素对应一个 3-bit code

    # ---- 三个 bit 平面 ----
    packed_b0 = pack_bits(((code >> 2) & 1).bool())
    packed_b1 = pack_bits(((code >> 1) & 1).bool())
    packed_b2 = pack_bits(( code       & 1).bool())

    # ---- 非离群：存 Sign(1)|Mantissa(7)；离群：存完整 BF16 + index ----
    non_outlier = (code != OUTLIER_CODE)
    packed_sm    = ((sign[non_outlier] << 7) | mant[non_outlier]).to(torch.uint8)

    outlier_mask = ~non_outlier
    outlier_idx  = torch.where(outlier_mask)[0].to(torch.int32)
    outlier_vals = weight.flatten()[outlier_mask]

    return {
        "shape":        weight.shape,
        "num_elem":     num_elem,
        "top_exps":     top_exps.to(torch.int32),
        "packed_b0":    packed_b0,
        "packed_b1":    packed_b1,
        "packed_b2":    packed_b2,
        "packed_sm":    packed_sm,
        "outlier_idx":  outlier_idx,
        "outlier_vals": outlier_vals,
    }


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------

def decode(compressed: dict) -> torch.Tensor:
    """从 TCA-TBE 编码还原 BF16 权重张量（bit-wise 无损）。"""
    num_elem = compressed["num_elem"]
    top_exps = compressed["top_exps"]

    # ---- 还原 code（三个 bit 平面合并）----
    b0 = unpack_bits(compressed["packed_b0"], num_elem)
    b1 = unpack_bits(compressed["packed_b1"], num_elem)
    b2 = unpack_bits(compressed["packed_b2"], num_elem)
    code = (b0.to(torch.uint8) << 2) | \
           (b1.to(torch.uint8) << 1) | \
           (b2.to(torch.uint8))

    # ---- 还原指数（从 code 反查 top_exps；离群位置后续直接覆盖，无需处理）----
    device = compressed["packed_b0"].device
    exp_recon = torch.zeros(num_elem, dtype=torch.int32, device=device)
    for i, e in enumerate(top_exps.tolist()):
        exp_recon[code == i] = e

    # ---- 还原 Sign / Mantissa（仅非离群元素，由 code != OUTLIER_CODE 判断）----
    non_outlier = (code != OUTLIER_CODE)
    packed_sm   = compressed["packed_sm"]

    sign_recon = torch.zeros(num_elem, dtype=torch.int32, device=device)
    mant_recon = torch.zeros(num_elem, dtype=torch.int32, device=device)
    sign_recon[non_outlier] = (packed_sm.to(torch.int32) >> 7) & 1
    mant_recon[non_outlier] =  packed_sm.to(torch.int32) & 0x7F

    # ---- 合并三个分量，再填入离群值 ----
    result_flat = components_to_bf16(sign_recon, exp_recon, mant_recon)

    outlier_idx  = compressed["outlier_idx"]
    outlier_vals = compressed["outlier_vals"]
    if outlier_idx.numel() > 0:
        result_flat[outlier_idx] = outlier_vals

    return result_flat.reshape(compressed["shape"])


# ---------------------------------------------------------------------------
# 压缩后实际字节数（不含调试辅助数据）
# ---------------------------------------------------------------------------

def compressed_bytes(compressed: dict) -> int:
    keys = ("top_exps", "packed_b0", "packed_b1", "packed_b2",
            "packed_sm", "outlier_idx", "outlier_vals")
    total = sum(compressed[k].numel() * compressed[k].element_size() for k in keys)
    total += 4 * len(compressed["shape"]) + 8   # shape + num_elem 元信息
    return total


# ---------------------------------------------------------------------------
# 测试验证
# ---------------------------------------------------------------------------

def test_lossless():
    print("=== TCA-TBE 无损压缩验证 ===\n")
    torch.manual_seed(42)

    cases = [
        ("正态分布 (小方差, 模拟 LLM 权重)", torch.randn(256, 256) * 0.02),
        ("正态分布 (大方差)",                torch.randn(256, 256) * 2.0),
        ("均匀分布",                         torch.rand(256, 256) - 0.5),
        ("全零",                             torch.zeros(256, 256)),
        ("单元素",                           torch.tensor([3.14])),
    ]

    for name, data_f32 in cases:
        data = data_f32.to(torch.bfloat16)
        orig_bytes = data.numel() * 2

        comp  = encode(data)
        comp_b = compressed_bytes(comp)
        ratio  = orig_bytes / comp_b

        recon = decode(comp)

        # bit-wise 验证
        ok = torch.all(data.view(torch.int16) == recon.view(torch.int16)).item()
        outlier_pct = comp["outlier_idx"].numel() / data.numel() * 100

        print(f"[{name}]")
        print(f"  原始: {orig_bytes/1024:.1f} KB  压缩: {comp_b/1024:.1f} KB  "
              f"压缩率: {ratio:.2f}x  离群: {outlier_pct:.2f}%  无损: {'PASS' if ok else 'FAIL'}")
        assert ok, f"bit-wise 验证失败：{name}"

    print("\n全部通过，bit-wise 无损。")


if __name__ == "__main__":
    test_lossless()
