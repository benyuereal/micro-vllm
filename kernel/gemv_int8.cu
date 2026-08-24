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
