"""融合 Gumbel-max 采样：argmax(logits/temp + gumbel_noise)，单趟 bf16 直读。

替代 sampler._gumbel_sample 的 torch.compile 路径（logits.float() 物化 311MB fp32
+ softmax + exponential + argmax，HBM ~1.4GB/step @ bs=512）。
Gumbel-max 定理：argmax_i (logit_i/temp + g_i)，g_i = -ln(-ln(u_i)), u_i~U(0,1)，
等价于按 softmax(logits/temp) 分布采样——无需显式 softmax/probs。

两趟：
  pass1 grid (bs*NUM_CHUNKS)：每 chunk 求局部 max+idx → [bs, NUM_CHUNKS]
  pass2 grid (bs)：跨 chunk argmax → token id
HBM 流量仅 logits 读一遍（155MB bf16 @ bs=512），理论 ~150us vs 现 ~930us。
"""
import torch
import triton
import triton.language as tl

VOCAB = 151936  # Qwen3-0.6B；其他 vocab 由调用方传参
CHUNK = 8192
NUM_CHUNKS = (VOCAB + CHUNK - 1) // CHUNK  # 19


@triton.jit
def _gumbel_chunk_kernel(LOGITS, TEMP, OUT_VAL, OUT_IDX, SEED,
                         stride_row, vocab: tl.constexpr,
                         CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS
    temp = tl.load(TEMP + row)
    base = row * stride_row + chunk * CHUNK
    offs = tl.arange(0, CHUNK)
    col = chunk * CHUNK + offs
    mask = col < vocab
    x = tl.load(LOGITS + base + offs, mask=mask, other=0.0).to(tl.float32)
    # gumbel 噪声：u~U(0,1)。xorshift-multiply hash（比 tl.rand 的 Philox 便宜得多，
    # 155MB 下 Philox 是瓶颈）。种子 = 元素地址 ^ SEED：SEED 每步变化 → 噪声每步 i.i.d.
    #（Gumbel-max 要求每步新噪声，固定噪声会让采样退化为确定性函数并偏置分布）。
    h = ((base + offs) ^ SEED).to(tl.int32)
    h ^= h >> 16
    h = (h * 0x7feb352d).to(tl.int32)
    h ^= h >> 15
    h = (h * -2073254261).to(tl.int32)  # 0x846ca68b 的 int32 表示
    h ^= h >> 16
    u = (h.to(tl.uint32) * (1.0 / 4294967296.0)).to(tl.float32)
    u = tl.clamp(u, 1e-12, 1.0 - 1e-12)
    g = -tl.log(-tl.log(u))
    v = x / temp + g
    v = tl.where(mask, v, -float("inf"))
    m = tl.max(v, axis=0)
    idx = tl.argmax(v, axis=0)
    tl.store(OUT_VAL + row * NUM_CHUNKS + chunk, m)
    tl.store(OUT_IDX + row * NUM_CHUNKS + chunk, (chunk * CHUNK + idx).to(tl.int32))


@triton.jit
def _gumbel_reduce_kernel(VAL, IDX, OUT, NUM_CHUNKS: tl.constexpr,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < NUM_CHUNKS
    v = tl.load(VAL + row * NUM_CHUNKS + offs, mask=mask, other=-float("inf"))
    sel = tl.argmax(v, axis=0)
    tok = tl.load(IDX + row * NUM_CHUNKS + sel)
    tl.store(OUT + row, tok)


class GumbelSampler:
    """预分配中间 buffer，__call__ 零分配。vocab 固定（Qwen3 151936）。"""

    def __init__(self, max_batch: int, device: str, vocab: int = VOCAB):
        self.vocab = vocab
        self.num_chunks = (vocab + CHUNK - 1) // CHUNK
        self._max_alloc = max_batch
        self._val = torch.empty(max_batch, self.num_chunks, dtype=torch.float32, device=device)
        self._idx = torch.empty(max_batch, self.num_chunks, dtype=torch.int32, device=device)
        self._out = torch.empty(max_batch, dtype=torch.int32, device=device)
        self._block = triton.next_power_of_2(self.num_chunks)

    def __call__(self, logits: torch.Tensor, temp: torch.Tensor, seed: int = 0) -> torch.Tensor:
        bs = logits.shape[0]
        if bs > self._max_alloc:  # 扩容（engine max_batch 可能超 init 值）
            self._val = torch.empty(bs, self.num_chunks, dtype=torch.float32, device=logits.device)
            self._idx = torch.empty(bs, self.num_chunks, dtype=torch.int32, device=logits.device)
            self._out = torch.empty(bs, dtype=torch.int32, device=logits.device)
            self._max_alloc = bs
        _gumbel_chunk_kernel[(bs * self.num_chunks,)](
            logits, temp, self._val, self._idx, seed, logits.stride(0),
            self.vocab, CHUNK, self.num_chunks, num_warps=8)
        _gumbel_reduce_kernel[(bs,)](
            self._val, self._idx, self._out, self.num_chunks, self._block)
        return self._out[:bs].clone()  # clone 避免返回内部 buffer 视图（防跨步 aliasing）
