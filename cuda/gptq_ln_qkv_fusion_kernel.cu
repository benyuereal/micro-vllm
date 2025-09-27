#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <math.h>

// vLLM风格的分块大小和常量
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// 最优的LayerNorm实现：使用Welford算法进行数值稳定的方差计算
__forceinline__ __device__ void compute_layernorm_stats(
    const half* input, int seq_idx, int hidden_dim, 
    float& mean, float& var, float eps) {
    
    // 使用Welford算法计算均值和方差（数值稳定）
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = 0; i < hidden_dim; i++) {
        float val = __half2float(input[seq_idx * hidden_dim + i]);
        sum += val;
        sum_sq += val * val;
    }
    
    mean = sum / hidden_dim;
    var = (sum_sq / hidden_dim) - (mean * mean);
    var = fmaxf(var, eps); // 确保方差不为负
}

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

// 完整的GPTQ 4bit解量化（参考vLLM实现）
__forceinline__ __device__ void dequant_4bit_8_gptq(
    uint32_t qw, uint32_t qz, half2 (&dq)[4], const float scale) {
  // 提取4bit值并反量化
  for (int i = 0; i < 4; i++) {
    int w = (qw >> (i * 8)) & 0xFF;
    int z = (qz >> (i * 8)) & 0xFF;
    
    // GPTQ解量化：weight = (qweight - qzeros) * scale
    half2 w01 = __halves2half2(
        __int2half_rn((w & 0xF) - (z & 0xF)), 
        __int2half_rn((w >> 4) - (z >> 4))
    );
    
    // 应用缩放
    dq[i] = __hmul2(w01, __float2half2_rn(scale));
  }
}

// 优化的融合LayerNorm + GPTQ QKV投影内核
template <int m_count>
__global__ void fused_ln_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size * seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight,     // GPTQ量化权重 [hidden_dim//8, hidden_dim*3]
    const uint32_t* __restrict__ qzeros,      // GPTQ量化零点 [num_groups, groupsize//8]
    const half* __restrict__ scales,          // GPTQ量化缩放 [num_groups, hidden_dim*3]
    const half* __restrict__ ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* __restrict__ ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* __restrict__ qkv_output,            // QKV输出 [batch_size * seq_len, hidden_dim * 3]
    const int batch_size, const int seq_len, const int hidden_dim,
    const int groupsize, const float eps) {
  
  auto t = threadIdx.x;
  
  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;
  
  int end_k = min(offset_k + BLOCK_KN_SIZE, hidden_dim);
  
  int n = offset_n + t * 4;
  
  // 共享内存用于LayerNorm计算
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];
  __shared__ float shared_mean[m_count];
  __shared__ float shared_var[m_count];
  
  // 1. 加载输入数据到共享内存
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* input_ptr = input + (offset_m + m) * hidden_dim;
      block_a[m][t] = input_ptr[offset_k + t];
    }
  }
  
  __syncthreads();
  
  // 2. 计算LayerNorm的均值和方差（每个序列独立计算）
  if (t == 0) {
    for (int m = 0; m < m_count; ++m) {
      compute_layernorm_stats(
          input, offset_m + m, hidden_dim, 
          shared_mean[m], shared_var[m], eps
      );
    }
  }
  
  __syncthreads();
  
  // 3. 应用LayerNorm归一化
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      float normalized = (__half2float(block_a[m][t]) - shared_mean[m]) / 
                        sqrtf(shared_var[m]);
      
      // 应用LayerNorm权重和偏置
      normalized = normalized * __half2float(ln_weight[offset_k + t]) + 
                   __half2float(ln_bias[offset_k + t]);
      
      block_a[m][t] = __float2half(normalized);
    }
  }
  
  __syncthreads();
  
  // 4. 零初始化输出
  if (n >= hidden_dim * 3) return;  // QKV总维度
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++) {
      int output_idx = (offset_m + m) * hidden_dim * 3 + n;
      if (output_idx < batch_size * seq_len * hidden_dim * 3) {
        *((uint64_t*)(qkv_output + output_idx)) = 0;
      }
    }
  }
  
  __syncthreads();
  
  // 5. GPTQ解量化和矩阵乘法（参考vLLM实现）
  int num_groups = hidden_dim / groupsize;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;
  
  // a, b offset
  int qk = offset_k / (32 / 4);
  
  const uint32_t* b_ptr = qweight + qk * (hidden_dim * 3) + n;
  const uint32_t* z_ptr = qzeros + group * (groupsize / 8) + (offset_k % groupsize) / 8;
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
      nextgroup += groupsize;
      z_ptr = qzeros + group * (groupsize / 8) + (k % groupsize) / 8;
      for (int i = 0; i < 4; i++) {
        scales_qkv[i] = __half2float(scales[group * (hidden_dim * 3) + n + i]);
      }
    }
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      const int4* z_ptr4 = (int4*)z_ptr;
      int4 load_int4 = *b_ptr4;
      int4 load_zeros = *z_ptr4;
      
      half2 dq[4][4];
      // 使用完整的GPTQ解量化（包含qzeros处理）
      dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_qkv[0]);
      dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_qkv[1]);
      dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_qkv[2]);
      dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_qkv[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c[m][0], 1.0f);
        block_c[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c[m][1], 1.0f);
        block_c[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c[m][2], 1.0f);
        block_c[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c[m][3], 1.0f);
      }
      
      b_ptr += (hidden_dim * 3);
      z_ptr += (groupsize / 8);
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 6. 输出QKV到正确的格式
  for (int m = 0; m < m_count; m++) {
    int output_base = (offset_m + m) * hidden_dim * 3 + n;
    
    if (output_base < batch_size * seq_len * hidden_dim * 3) {
      half2* qkv_out = (half2*)(qkv_output + output_base);
      half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
      half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
      atomicAdd(qkv_out, result01);
      atomicAdd(qkv_out + 1, result23);
    }
  }
}

// PyTorch包装函数
torch::Tensor fused_ln_qkv_gptq_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int groupsize,
    float eps
) {
    // 检查输入张量
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor [batch_size, seq_len, hidden_dim]");
    
    // 确保张量是连续的
    input = input.contiguous();
    qweight = qweight.contiguous();
    qzeros = qzeros.contiguous();
    scales = scales.contiguous();
    ln_weight = ln_weight.contiguous();
    ln_bias = ln_bias.contiguous();
    
    // 获取张量数据指针（使用vLLM相同的方法）
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    const uint32_t* qweight_ptr = qweight.data_ptr<uint32_t>();
    const uint32_t* qzeros_ptr = qzeros.data_ptr<uint32_t>();
    const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    const half* ln_weight_ptr = reinterpret_cast<const half*>(ln_weight.data_ptr<at::Half>());
    const half* ln_bias_ptr = reinterpret_cast<const half*>(ln_bias.data_ptr<at::Half>());
    
    // 创建输出张量 [batch_size * seq_len, hidden_dim * 3]
    auto qkv_output = torch::zeros({batch_size * seq_len, hidden_dim * 3}, 
                                  torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    
    half* qkv_output_ptr = reinterpret_cast<half*>(qkv_output.data_ptr<at::Half>());
    
    // 调用原始C函数
    {
        // 优化的网格和块大小（基于vLLM）
        dim3 blockDim, gridDim;
        blockDim.x = BLOCK_KN_SIZE;
        blockDim.y = 1;
        blockDim.z = 1;
        gridDim.x = DIVIDE(hidden_dim * 3, BLOCK_KN_SIZE * 4);  // QKV总维度
        gridDim.y = DIVIDE(batch_size * seq_len, BLOCK_M_SIZE_MAX);
        gridDim.z = DIVIDE(hidden_dim, BLOCK_KN_SIZE);
        
        // 启动融合内核（使用CUDA流，与vLLM一致）
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        fused_ln_qkv_gptq_kernel<1><<<gridDim, blockDim, 0, stream>>>(
            input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, ln_weight_ptr, ln_bias_ptr,
            qkv_output_ptr, batch_size, seq_len, hidden_dim, groupsize, eps
        );
        
        cudaDeviceSynchronize();
    }
    
    // 分割QKV并重塑为正确的格式
    auto q = qkv_output.slice(1, 0, hidden_dim).view({batch_size, seq_len, hidden_dim});
    auto k = qkv_output.slice(1, hidden_dim, hidden_dim*2).view({batch_size, seq_len, hidden_dim});
    auto v = qkv_output.slice(1, hidden_dim*2).view({batch_size, seq_len, hidden_dim});
    
    // 返回QKV张量的元组
    return torch::stack({q, k, v}, 0);
}

// PyTorch模块导出函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_qkv_gptq_cuda", &fused_ln_qkv_gptq_cuda, "Fused LayerNorm + QKV GPTQ CUDA Kernel");
}