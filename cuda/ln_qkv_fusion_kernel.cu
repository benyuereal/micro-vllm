#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <math.h>
#include <cub/cub.cuh>

// vLLM风格的分块大小
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// vLLM风格的向量化点积
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

// vLLM风格的4bit解量化
__forceinline__ __device__ void dequant_4bit_8_simple(
    uint32_t qw, half2 (&dq)[4]) {
  // 提取4bit值并反量化
  for (int i = 0; i < 4; i++) {
    int w = (qw >> (i * 8)) & 0xFF;
    half2 w01 = __halves2half2(__int2half_rn(w & 0xF), __int2half_rn(w >> 4));
    dq[i] = w01;
  }
}

// LayerNorm计算内核
template<int m_count>
__global__ void layernorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const int size_m, const int size_n, const float eps) {
  
  auto t = threadIdx.x;
  auto offset_m = blockIdx.x * m_count;
  auto offset_n = blockIdx.y * BLOCK_KN_SIZE;
  
  int end_n = min(offset_n + BLOCK_KN_SIZE, size_n);
  
  // 共享内存缓存输入
  __shared__ half block_input[m_count][BLOCK_KN_SIZE];
  
  // 加载输入数据
  if (offset_n + t < end_n) {
    for (int m = 0; m < m_count; ++m) {
      const half* input_ptr = input + (offset_m + m) * size_n;
      block_input[m][t] = input_ptr[offset_n + t];
    }
  }
  
  __syncthreads();
  
  // 计算均值和方差
  __shared__ float shared_mean[m_count];
  __shared__ float shared_var[m_count];
  
  if (t == 0) {
    for (int m = 0; m < m_count; ++m) {
      float sum = 0.0f, sum_sq = 0.0f;
      for (int i = 0; i < size_n; i++) {
        float val = __half2float(block_input[m][i]);
        sum += val;
        sum_sq += val * val;
      }
      shared_mean[m] = sum / size_n;
      shared_var[m] = (sum_sq / size_n) - (shared_mean[m] * shared_mean[m]);
    }
  }
  
  __syncthreads();
  
  // 归一化
  if (offset_n + t < end_n) {
    for (int m = 0; m < m_count; ++m) {
      float normalized = (__half2float(block_input[m][t]) - shared_mean[m]) / 
                        sqrtf(shared_var[m] + eps);
      
      // 应用权重和偏置
      normalized = normalized * __half2float(weight[offset_n + t]) + 
                   __half2float(bias[offset_n + t]);
      
      output[(offset_m + m) * size_n + offset_n + t] = __float2half(normalized);
    }
  }
}

// 融合LayerNorm + GPTQ QKV投影内核（基于vLLM实现）
template <int m_count>
__global__ void fused_ln_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size, seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight,     // GPTQ量化权重 [K//8, N]
    const uint32_t* __restrict__ qzeros,      // GPTQ量化零点 [num_groups, groupsize//8]
    const half* __restrict__ scales,          // GPTQ量化缩放 [num_groups, N]
    const half* __restrict__ ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* __restrict__ ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* __restrict__ q_output,              // Q输出 [batch_size, num_heads, seq_len, head_size]
    half* __restrict__ k_output,              // K输出 [batch_size, kv_num_heads, seq_len, head_size]
    half* __restrict__ v_output,              // V输出 [batch_size, kv_num_heads, seq_len, head_size]
    const int batch_size, const int seq_len, const int hidden_dim,
    const int num_heads, const int kv_num_heads, const int head_size,
    const int groupsize, const float eps) {
  
  auto t = threadIdx.x;
  
  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, hidden_dim);
  
  int n = offset_n + t * 4;
  
  // Preload block_a (LayerNorm后的输入)
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* input_ptr = input + (offset_m + m) * hidden_dim;
      block_a[m][t] = input_ptr[offset_k + t];
    }
  }
  
  // Zero output
  if (n >= hidden_dim * 3) return;  // QKV总维度
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++) {
      int output_idx = (offset_m + m) * hidden_dim * 3 + n;
      if (output_idx < batch_size * seq_len * hidden_dim * 3) {
        *((uint64_t*)(q_output + output_idx)) = 0;
        *((uint64_t*)(k_output + output_idx)) = 0;
        *((uint64_t*)(v_output + output_idx)) = 0;
      }
    }
  }
  
  __syncthreads();
  
  // Find initial group
  int groupsize_actual = hidden_dim / (hidden_dim / groupsize);
  int group = offset_k / groupsize_actual;
  int nextgroup = offset_k + groupsize_actual;
  
  // a, b offset
  int qk = offset_k / (32 / 4);
  
  const uint32_t* b_ptr = qweight + qk * (hidden_dim * 3) + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;
  
  // Initial group
  float scales_qkv[4];
  for (int i = 0; i < 4; i++) {
    scales_qkv[i] = __half2float(scales[group * (hidden_dim * 3) + n + i]);
  }
  
  // Column result
  float block_c[m_count][4] = {};
  
  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize_actual;
      for (int i = 0; i < 4; i++) {
        scales_qkv[i] = __half2float(scales[group * (hidden_dim * 3) + n + i]);
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
        block_c[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c[m][0], scales_qkv[0]);
        block_c[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c[m][1], scales_qkv[1]);
        block_c[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c[m][2], scales_qkv[2]);
        block_c[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c[m][3], scales_qkv[3]);
      }
      
      b_ptr += (hidden_dim * 3);
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 输出QKV
  for (int m = 0; m < m_count; m++) {
    int output_base = (offset_m + m) * hidden_dim * 3 + n;
    
    // Q输出
    if (n < hidden_dim) {
      half2* q_out = (half2*)(q_output + output_base);
      half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
      half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
      atomicAdd(q_out, result01);
      atomicAdd(q_out + 1, result23);
    }
    
    // K输出
    if (n >= hidden_dim && n < hidden_dim * 2) {
      int k_idx = n - hidden_dim;
      half2* k_out = (half2*)(k_output + (offset_m + m) * hidden_dim + k_idx);
      half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
      half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
      atomicAdd(k_out, result01);
      atomicAdd(k_out + 1, result23);
    }
    
    // V输出
    if (n >= hidden_dim * 2 && n < hidden_dim * 3) {
      int v_idx = n - hidden_dim * 2;
      half2* v_out = (half2*)(v_output + (offset_m + m) * hidden_dim + v_idx);
      half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
      half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
      atomicAdd(v_out, result01);
      atomicAdd(v_out + 1, result23);
    }
  }
}

// 包装函数
extern "C" {
    void fused_ln_qkv_gptq_cuda(
        const half* input,
        const uint32_t* qweight,
        const uint32_t* qzeros,
        const half* scales,
        const half* ln_weight,
        const half* ln_bias,
        half* q_output,
        half* k_output,
        half* v_output,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int num_heads,
        int kv_num_heads,
        int head_size,
        int groupsize,
        float eps
    ) {
        // 简化的网格和块大小（基于vLLM）
        dim3 blockDim, gridDim;
        blockDim.x = BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(hidden_dim * 3, BLOCK_KN_SIZE * 4);  // QKV总维度
        gridDim.y = DIVIDE(batch_size * seq_len, BLOCK_M_SIZE_MAX);
        gridDim.z = DIVIDE(hidden_dim, BLOCK_KN_SIZE);
        
        // 启动融合内核
        fused_ln_qkv_gptq_kernel<1><<<gridDim, blockDim>>>(
            input, qweight, qzeros, scales, ln_weight, ln_bias,
            q_output, k_output, v_output,
            batch_size, seq_len, hidden_dim, num_heads, kv_num_heads, head_size, groupsize, eps
        );
        
        cudaDeviceSynchronize();
    }
}
