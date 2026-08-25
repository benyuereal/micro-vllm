// W8A16 int8 GEMV：单用户 decode（M=1）权重 INT8 + 激活 bf16。
//
// 布局：W_int8[N,K] int8 连续（每输出行连续 K），scale[N] fp32（per-output-row）。
//   out[m,n] = scale[n] * sum_k x[m,k] * w_int8[n,k]
// 每 warp（32 thread）算一个 (m, row) 输出：int8 向量化读 W 行（连续）与 X 行，
// fp32 累加，warp reduce 后乘 scale[row] 写 bf16。
//
// 相比 bf16 GEMV：权重字节数减半（int8 vs bf16）→ decode memory-bound 下带宽减半。
// int8 读用 int4（16 字节 = 16 int8）向量化，比 bf162（4 字节）更宽。

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void gemv_int8_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ SCALE,
    __nv_bfloat16* __restrict__ OUT,
    int K, int N, int M)
{
    int m = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || m >= M) return;

    const __nv_bfloat16* xrow = X + m * K;
    const int8_t* wrow = W + (long)row * K;   // W 行连续 K → coalesced
    const __nv_bfloat162* xv = (const __nv_bfloat162*)xrow;
    const int4* wv = (const int4*)wrow;       // int4 = 16 int8

    int K16 = K / 16;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K16; i += 32) {
        int4 wpack = wv[i];
        const int8_t* wp = (const int8_t*)&wpack;
        // 对应 16 个 x = xv[i*8 .. i*8+7]（8 个 bf162）
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float2 xf = __bfloat1622float2(xv[i * 8 + j]);
            acc += xf.x * (float)wp[2 * j] + xf.y * (float)wp[2 * j + 1];
        }
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (threadIdx.x == 0)
        OUT[m * N + row] = __float2bfloat16(acc * SCALE[row]);
}

// group-128 int8 GEMV（Qwen3.8 W8A16 预量化格式）：
//   out[m,n] = sum_g scale[n,g] * (sum_{k in group g} x[m,k] * w_int8[n,k])
// scale [N, K/128] fp32（group-128 对称量化）。
// 每 warp 算一个 (m,row) 输出；每 lane 负责一个 128 元素 group（8 个 int4=128 int8），
// fp32 累加该 group 的 x·w，乘本 lane 的 scale[n,g]，再 warp reduce 求和（各 lane 是
// 不同 group，scale 不同，故先各自乘 scale 再 reduce）。
// 访存：lane t 读 wrow[(g0+t)*128 .. +127]，相邻 lane 读相邻 128B → coalesced。
__global__ void gemv_int8_group_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ SCALE,   // [N, K/128]
    __nv_bfloat16* __restrict__ OUT,
    int K, int N, int M, int NGROUPS)  // NGROUPS = K/128
{
    int m = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || m >= M) return;

    const __nv_bfloat16* xrow = X + m * K;
    const int8_t* wrow = W + (long)row * K;
    const float* srow = SCALE + (long)row * NGROUPS;

    float total = 0.f;
    // 每 lane 一个 group（128 int8 = 8 int4）；32 lane = 32 group/iter。
    for (int g0 = 0; g0 < NGROUPS; g0 += 32) {
        int g = g0 + threadIdx.x;
        float acc = 0.f;
        if (g < NGROUPS) {
            const int4* wv = (const int4*)(wrow + g * 128);
            const __nv_bfloat162* xv = (const __nv_bfloat162*)(xrow + g * 128);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int4 wpack = wv[j];
                const int8_t* wp = (const int8_t*)&wpack;
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    float2 xf = __bfloat1622float2(xv[j * 8 + jj]);
                    acc += xf.x * (float)wp[2 * jj] + xf.y * (float)wp[2 * jj + 1];
                }
            }
            acc *= srow[g];   // 本 lane 的 group scale（各 lane 不同）
        }
        // warp reduce：各 lane 是不同 group，求和即 total 的该 iter 贡献
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (threadIdx.x == 0) total += acc;
    }
    if (threadIdx.x == 0)
        OUT[m * N + row] = __float2bfloat16(total);
}

// =====================================================================
// 分块 int8 GEMV（group-128）：小 M（投机解码 verify，M=1+N≈8）权重 HBM 只读一次。
//
// 与 gemv_int8_group_kernel 的【逐 (m,row) fp32 累加顺序完全一致】（bit-exact）：
//   每 (m,row)：total=0；for g0 in 0,32,...：acc=本 lane group 的 x·w（j/jj 顺序同
//   原 kernel）*scale；warp reduce（16→1 树）；lane0 total+=acc。
// 区别仅在访存：原 kernel 每 (m,row) 一个 warp（grid.y=M），W[row] 被 M 个 warp 各读
// 一遍（HBM 1 次 + L2 M-1 次）；本 kernel 每 row 一个 warp，W[row] 分块载入 shared
// （HBM 只读 1 次），M 个 m 复用 shared。M=8 时比原 GEMV 快 ~8x（权重读 8→1 次）。
//
// 为何不用 TileLang bf16 GEMM：GEMM 先把 int8 反量化成 bf16（舍入），与 decode 的
// int8 精确累加（fp32）数值不同，greedy 近并列处 argmax 会翻转 → 投机解码输出与
// 非 spec 不逐 token 一致。本 kernel 保持 int8 精确累加，bit-exact。
//
// 限制：M<=8（verify M=1+N，N<=7）。M>8 由 wrapper 回退原 GEMV。
#define TILED_ROWS 8
#define TILED_CHUNK_GROUPS 32
#define TILED_CHUNK_INT8 (TILED_CHUNK_GROUPS * 128)   // 4096

__global__ void gemv_int8_group_tiled_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ SCALE,   // [N, K/128]
    __nv_bfloat16* __restrict__ OUT,
    int K, int N, int M, int NGROUPS)
{
    __shared__ int8_t w_shared[TILED_ROWS][TILED_CHUNK_INT8];
    __shared__ float s_shared[TILED_ROWS][TILED_CHUNK_GROUPS];
    __shared__ float total_shared[TILED_ROWS][8];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * TILED_ROWS + warp;
    if (row >= N) return;

    const int8_t* wrow = W + (long)row * K;
    const float* srow = SCALE + (long)row * NGROUPS;

    if (lane == 0)
        for (int m = 0; m < 8; m++) total_shared[warp][m] = 0.f;
    __syncwarp();

    for (int g0 = 0; g0 < NGROUPS; g0 += TILED_CHUNK_GROUPS) {
        int ngroups = min(TILED_CHUNK_GROUPS, NGROUPS - g0);
        int nint8 = ngroups * 128;
        // W[row, g0*128 : +nint8] 载入 shared（int4=16B 向量化，coalesced，HBM 只读一次）。
        // 原逐字节载入（32B 事务）→ 改 int4（512B/iter，4×128B 事务），减少 W 载入 stall。
        int4* ws4 = (int4*)w_shared[warp];
        const int4* wr4 = (const int4*)(wrow + g0 * 128);
        int nint4 = nint8 / 16;
        for (int i = lane; i < nint4; i += 32)
            ws4[i] = wr4[i];
        for (int i = lane; i < ngroups; i += 32)
            s_shared[warp][i] = srow[g0 + i];
        __syncwarp();

        for (int m = 0; m < M; m++) {
            const __nv_bfloat16* xrow = X + (long)m * K;
            float acc = 0.f;
            int g = g0 + lane;
            if (g < NGROUPS) {
                const int8_t* ws = w_shared[warp] + (g - g0) * 128;
                const __nv_bfloat162* xv = (const __nv_bfloat162*)(xrow + g * 128);
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int4 wpack = *(const int4*)(ws + j * 16);
                    const int8_t* wp = (const int8_t*)&wpack;
                    #pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        float2 xf = __bfloat1622float2(xv[j * 8 + jj]);
                        acc += xf.x * (float)wp[2 * jj] + xf.y * (float)wp[2 * jj + 1];
                    }
                }
                acc *= s_shared[warp][g - g0];
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) total_shared[warp][m] += acc;
        }
        __syncwarp();
    }
    for (int m = 0; m < M; m++)
        if (lane == 0) OUT[(long)m * N + row] = __float2bfloat16(total_shared[warp][m]);
}

// out = x @ w_int8.t()（group-128，分块，权重 HBM 只读一次，bit-exact 原 GEMV）。
// x:[M,K] bf16, w_int8:[N,K] int8, scale:[N,K/128] fp32, out:[M,N] bf16。M<=8。
torch::Tensor gemv_int8_group_tiled(torch::Tensor x, torch::Tensor w_int8,
                                    torch::Tensor scale, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(x.is_cuda() && w_int8.is_cuda() && scale.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    int M = x.size(0);
    int K = x.size(1);
    int N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "w_int8 second dim must equal x second dim (K)");
    TORCH_CHECK(scale.size(0) == N && scale.size(1) == K / 128, "scale must be [N, K/128]");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape must be [M,N]");
    TORCH_CHECK(K % 128 == 0, "K must be multiple of 128 (group size)");
    TORCH_CHECK(M <= 8, "tiled GEMV requires M<=8");

    dim3 grid((N + TILED_ROWS - 1) / TILED_ROWS);
    dim3 block(32 * TILED_ROWS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_int8_group_tiled_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (const int8_t*)w_int8.data_ptr(),
        (const float*)scale.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),
        K, N, M, K / 128);
    return out;
}

// out = x @ w_int8.t()（group-128）。x:[M,K] bf16, w_int8:[N,K] int8,
// scale:[N,K/128] fp32, out:[M,N] bf16。
torch::Tensor gemv_int8_group(torch::Tensor x, torch::Tensor w_int8,
                              torch::Tensor scale, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(x.is_cuda() && w_int8.is_cuda() && scale.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    int M = x.size(0);
    int K = x.size(1);
    int N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "w_int8 second dim must equal x second dim (K)");
    TORCH_CHECK(scale.size(0) == N && scale.size(1) == K / 128, "scale must be [N, K/128]");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape must be [M,N]");
    TORCH_CHECK(K % 128 == 0, "K must be multiple of 128 (group size)");

    const int rpb = 4;
    dim3 grid((N + rpb - 1) / rpb, M);
    dim3 block(32, rpb);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_int8_group_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (const int8_t*)w_int8.data_ptr(),
        (const float*)scale.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),
        K, N, M, K / 128);
    return out;
}

// out = x @ w_int8.t() * scale。x:[M,K] bf16, w_int8:[N,K] int8, scale:[N] fp32, out:[M,N] bf16。
torch::Tensor gemv_int8(torch::Tensor x, torch::Tensor w_int8, torch::Tensor scale, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w_int8.scalar_type() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(x.is_cuda() && w_int8.is_cuda() && scale.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    int M = x.size(0);
    int K = x.size(1);
    int N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "w_int8 second dim must equal x second dim (K)");
    TORCH_CHECK(scale.size(0) == N, "scale must be [N]");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape must be [M,N]");
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16 for int4 vectorization");

    const int rpb = 4;
    dim3 grid((N + rpb - 1) / rpb, M);
    dim3 block(32, rpb);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_int8_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (const int8_t*)w_int8.data_ptr(),
        (const float*)scale.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),
        K, N, M);
    return out;
}
