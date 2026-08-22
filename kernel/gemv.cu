// 手写 CUDA GEMV：单用户 decode（M=1）替代 cuBLAS，快 32-44%。
//
// 布局：W_t[N,K] row-major 连续（prepare_weights 预存 .t().contiguous() 副本）。
//   Y[M,N] = X[M,K] @ W_t[N,K].t()   等价 out[m,row] = sum_k X[m,k] * W_t[row*K + k]
// 每 warp（32 thread）算一个 (m, row) 输出：bf162 向量化读 W_t 行（连续）与 X 行，
// shuffle reduce。grid=((N+rpb-1)/rpb, M), block=(32, rpb)，rpb=4。
//
// 设计依据（L20 sm_89, bf16, CUDA13 实测）：
//   - 1 warp per output row 是 M=1 最优（block-per-row 多线程同 row reduce 反而慢）。
//   - bf162 向量化 load（__nv_bfloat162）比标量快、比 float4 type-pun 稳（float4 在
//     CUDA13 触发 illegal memory access）。
//   - M=1 全胜 cuBLAS；M=2 多数胜；M>=4 cuBLAS 切 tensor-core GEMM 反超——故仅 M=1 用此 kernel。
//   - W 必须是 [N,K] 连续（GEMV 友好）。若 W 为 [K,N]，列跨步读不 coalesced，42us 灾难慢。
//
// persistent（48 block 串行循环处理行）= 95us 反例：GEMV 必须 grid=N/rpb 充分并行。

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// 通用 GEMV/GEMM（M=1 最优，M>1 可用但 M>=4 不如 cuBLAS）。
// X:[M,K] bf16, W_t:[N,K] bf16, OUT:[M,N] bf16。OUT 可等于预分配 buffer（in-place，graph 友好）。
__global__ void gemv_v2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_t,
    __nv_bfloat16* __restrict__ OUT,
    int K, int N, int M)
{
    int m = blockIdx.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N || m >= M) return;

    const __nv_bfloat16* xrow = X + m * K;
    const __nv_bfloat16* wrow = W_t + row * K;   // W_t 行连续 K → coalesced
    const __nv_bfloat162* xv = (const __nv_bfloat162*)xrow;
    const __nv_bfloat162* wv = (const __nv_bfloat162*)wrow;

    int K2 = K / 2;
    float acc = 0.f;
    for (int i = threadIdx.x; i < K2; i += 32) {
        __nv_bfloat162 a = xv[i], b = wv[i];
        float2 af = __bfloat1622float2(a), bf = __bfloat1622float2(b);
        acc += af.x * bf.x + af.y * bf.y;
    }
    // warp reduce
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (threadIdx.x == 0)
        OUT[m * N + row] = __float2bfloat16(acc);
}

// out = x @ w_t.t()。x:[M,K] w_t:[N,K] out:[M,N]，均 bf16 contiguous。
// rpb=rows-per-block（blockDim.y），4 实测最优；M 维走 grid.y。
torch::Tensor gemv_v2(torch::Tensor x, torch::Tensor w_t, torch::Tensor out) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16, "w_t must be bf16");
    TORCH_CHECK(x.is_cuda() && w_t.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    int M = x.size(0);
    int K = x.size(1);
    int N = w_t.size(0);
    TORCH_CHECK(w_t.size(1) == K, "w_t second dim must equal x second dim (K)");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape must be [M,N]");

    const int rpb = 4;
    dim3 grid((N + rpb - 1) / rpb, M);
    dim3 block(32, rpb);
    // 显式用当前 stream（at::cuda::getCurrentCUDAStream()）：CUDA graph capture 时
    // 必须在 capture stream 上 launch，否则 kernel 不被记入 graph（replay 时静默不执行，
    // 输出 buffer 保持未初始化 → logits 坍缩 0）。默认 <<<>>> 用 legacy default stream 会出此问题。
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    gemv_v2_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (const __nv_bfloat16*)w_t.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(),
        K, N, M);
    return out;
}
