#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

// vLLM风格的分块大小
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// 简化的向量化点积
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

// 简化的4bit反量化
__forceinline__ __device__ void dequant_4bit_8_simple(
    uint32_t qw, half2 (&dq)[4]) {
  // 提取4bit值并反量化
  for (int i = 0; i < 4; i++) {
    int w = (qw >> (i * 8)) & 0xFF;
    half2 w01 = __halves2half2(__int2half_rn(w & 0xF), __int2half_rn(w >> 4));
    dq[i] = w01;
  }
}

// vLLM风格内核
template <int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  
  auto t = threadIdx.x;
  
  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  
  int n = offset_n + t * 4;
  
  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a + (offset_m + m) * size_k;
      half* block_a_ptr = block_a[m];
      block_a_ptr[t] = a_ptr[offset_k + t];
    }
  }
  
  // Zero output
  if (n >= size_n) return;
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)(c + (offset_m + m) * size_n + n)) = 0;
  }
  
  __syncthreads();
  
  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;
  
  // a, b offset
  int qk = offset_k / (32 / 4);
  
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;
  
  // Initial group
  float scales[4];
  for (int i = 0; i < 4; i++) {
    scales[i] = __half2float(b_gptq_scales[group * size_n + n + i]);
  }
  
  // Column result
  float block_c[m_count][4] = {};
  
  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      for (int i = 0; i < 4; i++) {
        scales[i] = __half2float(b_gptq_scales[group * size_n + n + i]);
      }
    }
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;
      
      half2 dq[4][4];
      dequant_4bit_8_simple(load_int4.x, dq[0]);
      dequant_4bit_8_simple(load_int4.y, dq[1]);
      dequant_4bit_8_simple(load_int4.z, dq[2]);
      dequant_4bit_8_simple(load_int4.w, dq[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      
      b_ptr += size_n;
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)(c + (offset_m + m) * size_n + n);
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

// cuBLAS辅助函数
cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasCreate(&handle);
  }
  return handle;
}

// Python接口
torch::Tensor fused_gptq_gemm_4bit_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    int groupsize
) {
  // 获取维度
  int M = input.size(0);
  int K = input.size(1);
  int N = qweight.size(0);
  
  // 创建输出张量
  auto output = torch::zeros({M, N}, torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device()));
  
  // 简化的网格和块大小
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(N, BLOCK_KN_SIZE * 4);
  gridDim.y = DIVIDE(M, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(K, BLOCK_KN_SIZE);
  
  // 启动内核
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  gemm_half_q_half_gptq_4bit_kernel<1><<<gridDim, blockDim, 0, stream>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      qweight.data_ptr<uint32_t>(),
      qzeros.data_ptr<uint32_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      M, N, K, K / groupsize
  );
  
  cudaDeviceSynchronize();
  return output;
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gptq_gemm_4bit_cuda", &fused_gptq_gemm_4bit_cuda, "vLLM Style Fused GPTQ GEMM 4bit CUDA");
}
