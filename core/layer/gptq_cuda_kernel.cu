#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// vLLM风格的分块大小
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// vLLM风格的向量化点积
__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr,
                                           const float g_result,
                                           const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

// vLLM风格的4bit反量化
__forceinline__ __device__ void dequant_4bit_8_gptq(
    uint32_t qw, half2 (&dq)[4], const half2 (&z1z16)[2], 
    const half2 (&y1y16)[2], int size_n, bool alt) {
  // 提取4bit值并反量化
  for (int i = 0; i < 4; i++) {
    int w = (qw >> (i * 8)) & 0xFF;
    half2 w01 = __halves2half2(__int2half_rn(w & 0xF), __int2half_rn(w >> 4));
    dq[i] = __hfma2(w01, y1y16[i % 2], z1z16[i % 2]);
  }
}

// vLLM风格的主内核
template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  
  auto t = threadIdx.x;
  
  // 分块
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  int n = offset_n + t * 4;
  
  // 预加载block_a到共享内存
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      block_a[m][t] = a[(offset_m + m) * size_k + offset_k + t];
    }
  }
  
  // 零化输出
  if (n >= size_n) return;
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)&c[(offset_m + m) * size_n + n]) = 0;
  }
  
  __syncthreads();
  
  // 找到初始组
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;
  
  // 计算偏移
  int qk = offset_k / (32 / 4);
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;
  
  // 初始组
  int zeros[4];
  float scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  
  // 加载零点和缩放因子
  for (int i = 0; i < 4; i++) {
    zeros[i] = b_gptq_qzeros[group * (size_n / 8) + (n + i) / 8];
    scales[i] = __half2float(b_gptq_scales[group * size_n + n + i]);
  }
  
  // 预计算反量化参数
  for (int i = 0; i < 4; i++) {
    int zero_val = (zeros[i] >> ((n + i) % 8 * 4)) & 0xF;
    z1z16[i][0] = __halves2half2(__int2half_rn(-zero_val - 1), __int2half_rn(-zero_val - 1));
    z1z16[i][1] = __halves2half2(__int2half_rn(-zero_val - 1), __int2half_rn(-zero_val - 1));
    y1y16[i][0] = __halves2half2(__float2half(scales[i]), __float2half(scales[i]));
    y1y16[i][1] = __halves2half2(__float2half(scales[i]), __float2half(scales[i]));
  }
  
  // 列结果
  float block_c[m_count][4] = {};
  
  // 反量化和乘法
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      // 重新加载组参数
      for (int i = 0; i < 4; i++) {
        zeros[i] = b_gptq_qzeros[group * (size_n / 8) + (n + i) / 8];
        scales[i] = __half2float(b_gptq_scales[group * size_n + n + i]);
      }
    }
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;
      
      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0], block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1], block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2], block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3], block_c[m][3]);
      }
      
      b_ptr += size_n;
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 存储结果
  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)&c[(offset_m + m) * size_n + n];
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

// 内核选择函数
typedef void (*fp_gemm_half_q_half_gptq_kernel)(const half*, const uint32_t*,
                                                const uint32_t*, const half*,
                                                half*, const int, const int,
                                                const int, const int);

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel_wrapper(
    const half* a, const uint32_t* b_q_weight,
    const uint32_t* b_gptq_qzeros,
    const half* b_gptq_scales, half* c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  gemm_half_q_half_gptq_4bit_kernel<first_block, m_count><<<1, 1>>>(
      a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c, size_m, size_n, size_k, groups);
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(
    bool first_block, const int m_count) {
  if (m_count == 1) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 1>;
  if (m_count == 2) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 2>;
  if (m_count == 3) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 3>;
  if (m_count == 4) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 4>;
  if (m_count == 5) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 5>;
  if (m_count == 6) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 6>;
  if (m_count == 7) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 7>;
  if (m_count == 8) return gemm_half_q_half_gptq_4bit_kernel_wrapper<true, 8>;
  return NULL;
}

void gemm_half_q_half_cuda_part(const half* a, const uint32_t* b_q_weight,
                                const uint32_t* b_gptq_qzeros,
                                const half* b_gptq_scales, half* c,
                                int size_m, int size_n, int size_k,
                                int m_count, int groups) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
  gridDim.y = DIVIDE(size_m, m_count);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);
  
  fp_gemm_half_q_half_gptq_kernel kernel =
      pick_gemm_half_q_half_gptq_kernel(true, m_count);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(a, b_q_weight, b_gptq_qzeros,
                                           b_gptq_scales, c, size_m, size_n,
                                           size_k, groups);
}

// Python接口 - 适配你的代码格式
torch::Tensor fused_gptq_gemm_4bit_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    int groupsize
) {
  // 获取维度 - 适配你的格式
  int M = input.size(0);
  int K = input.size(1);
  int N = qweight.size(0);  // qweight=[N, K//8]
  int num_groups = scales.size(0);  // scales=[num_groups, K]
  
  // 创建输出张量
  auto output = torch::zeros({M, N}, torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device()));
  
  // 转换数据类型和格式以适配vLLM内核
  // 1. 转换qweight: [N, K//8] -> [K//8, N]
  auto qweight_transposed = qweight.t().contiguous();
  
  // 2. 转换qzeros: [num_groups, K//8] -> [num_groups, K//8] (保持不变)
  auto qzeros_contiguous = qzeros.contiguous();
  
  // 3. 转换scales: [num_groups, K] -> [num_groups, N] (需要重新排列)
  auto scales_reshaped = scales.contiguous();
  
  // 使用vLLM风格的分块策略
  int max_chunks = M / BLOCK_M_SIZE_MAX;
  int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
  int last_chunk_size = M - last_chunk;
  
  if (max_chunks) {
    gemm_half_q_half_cuda_part(
        input.data_ptr<at::Half>(),
        qweight_transposed.data_ptr<uint32_t>(),
        qzeros_contiguous.data_ptr<uint32_t>(),
        scales_reshaped.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        last_chunk, N, K, BLOCK_M_SIZE_MAX, num_groups
    );
  }
  
  if (last_chunk_size) {
    gemm_half_q_half_cuda_part(
        input.data_ptr<at::Half>() + last_chunk * K,
        qweight_transposed.data_ptr<uint32_t>(),
        qzeros_contiguous.data_ptr<uint32_t>(),
        scales_reshaped.data_ptr<at::Half>(),
        output.data_ptr<at::Half>() + last_chunk * N,
        last_chunk_size, N, K, last_chunk_size, num_groups
    );
  }
  
  cudaDeviceSynchronize();
  return output;
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gptq_gemm_4bit_cuda", &fused_gptq_gemm_4bit_cuda, "Fused GPTQ 4bit GEMM CUDA");
}
