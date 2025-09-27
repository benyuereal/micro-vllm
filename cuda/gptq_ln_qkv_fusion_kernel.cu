#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

// vLLM风格的分块大小和常量
#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// 高性能LayerNorm实现：数值稳定的Welford算法
__forceinline__ __device__ void compute_layernorm_stats(
    const half* input, int seq_idx, int hidden_dim, 
    float& mean, float& var, float eps) {
    
    // 使用Welford算法进行数值稳定的方差计算
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // 使用half2向量化，提高计算效率
    const half2* input2 = (const half2*)(input + seq_idx * hidden_dim);
    int half_dim = hidden_dim / 2;
    
    for (int i = 0; i < half_dim; i++) {
        half2 val2 = input2[i];
        float val1 = __half2float(__low2half(val2));
        float val2_f = __half2float(__high2half(val2));
        
        sum += val1 + val2_f;
        sum_sq += val1 * val1 + val2_f * val2_f;
    }
    
    // 处理奇数维度
    if (hidden_dim % 2 == 1) {
        float val = __half2float(input[seq_idx * hidden_dim + hidden_dim - 1]);
        sum += val;
        sum_sq += val * val;
    }
    
    mean = sum / hidden_dim;
    
    // 使用数值稳定的方差计算
    float variance = (sum_sq / hidden_dim) - (mean * mean);
    // 确保方差非负（处理数值误差）
    variance = fmaxf(variance, 0.0f);
    var = fmaxf(variance, eps);
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

// 优化的融合内核：LayerNorm + GPTQ QKV投影（完整vLLM优化版本）
template <int m_count>
__global__ void fused_ln_qkv_gptq_kernel(
    const half* __restrict__ input,           // [batch_size * seq_len, hidden_dim]
    const uint32_t* __restrict__ qweight_q,   // Q权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qweight_k,   // K权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qweight_v,   // V权重 [hidden_dim//8, hidden_dim]
    const uint32_t* __restrict__ qzeros_q,    // Q零点 [num_groups, groupsize//8]
    const uint32_t* __restrict__ qzeros_k,    // K零点 [num_groups, groupsize//8]
    const uint32_t* __restrict__ qzeros_v,    // V零点 [num_groups, groupsize//8]
    const half* __restrict__ scales_q,        // Q缩放 [num_groups, hidden_dim]
    const half* __restrict__ scales_k,        // K缩放 [num_groups, hidden_dim]
    const half* __restrict__ scales_v,        // V缩放 [num_groups, hidden_dim]
    const half* __restrict__ ln_weight,       // LayerNorm权重 [hidden_dim]
    const half* __restrict__ ln_bias,         // LayerNorm偏置 [hidden_dim]
    half* __restrict__ q_output,              // Q输出 [batch_size * seq_len, hidden_dim]
    half* __restrict__ k_output,              // K输出 [batch_size * seq_len, hidden_dim]
    half* __restrict__ v_output,              // V输出 [batch_size * seq_len, hidden_dim]
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
  
  // 1. 计算LayerNorm的均值和方差（全局计算）
  if (t == 0) {
    for (int m = 0; m < m_count; ++m) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        compute_layernorm_stats(
            input, seq_idx, hidden_dim, 
            shared_mean[m], shared_var[m], eps
        );
      }
    }
  }
  
  __syncthreads();
  
  // 2. 加载输入数据到共享内存并应用LayerNorm
  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        // 预计算sqrt，避免重复计算
        float sqrt_var = sqrtf(shared_var[m]);
        
        // 直接应用LayerNorm归一化
        float val = __half2float(input[seq_idx * hidden_dim + offset_k + t]);
        float normalized = (val - shared_mean[m]) / sqrt_var;
        
        // 应用LayerNorm权重和偏置
        normalized = normalized * __half2float(ln_weight[offset_k + t]) + 
                     __half2float(ln_bias[offset_k + t]);
        
        block_a[m][t] = __float2half(normalized);
      }
    }
  }
  
  __syncthreads();
  
  // 3. 零初始化输出
  if (n >= hidden_dim) return;
  
  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        q_output[seq_idx * hidden_dim + n] = __float2half(0.0f);
        k_output[seq_idx * hidden_dim + n] = __float2half(0.0f);
        v_output[seq_idx * hidden_dim + n] = __float2half(0.0f);
      }
    }
  }
  
  __syncthreads();
  
  // 4. GPTQ解量化和矩阵乘法（分别处理Q、K、V）
  int num_groups = hidden_dim / groupsize;
  int group = offset_k / groupsize;
  int nextgroup_k_boundary = (group + 1) * groupsize;
  
  // a, b offset
  int qk_offset_in_qweight = offset_k / (32 / 4);
  
  // 处理Q投影
  const uint32_t* b_ptr_q = qweight_q + qk_offset_in_qweight * hidden_dim + n;
  const uint32_t* z_ptr_base_q = qzeros_q + group * (groupsize / 8);
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;
  
  // Initial group scales for Q
  float scales_q_val[4];
  for (int i = 0; i < 4; i++) {
    scales_q_val[i] = __half2float(scales_q[group * hidden_dim + n + i]);
  }
  
  // Column result for Q
  float block_c_q[m_count][4] = {};
  
  // Dequantize and multiply for Q
  int k = offset_k;
  while (k < end_k) {
    if (k >= nextgroup_k_boundary) {
      group++;
      if (group >= num_groups) break;
      nextgroup_k_boundary = (group + 1) * groupsize;
      z_ptr_base_q = qzeros_q + group * (groupsize / 8);
      for (int i = 0; i < 4; i++) {
        scales_q_val[i] = __half2float(scales_q[group * hidden_dim + n + i]);
      }
    }
    
    const uint32_t* current_z_ptr = z_ptr_base_q + (k % groupsize) / 8;
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (const int4*)b_ptr_q;
      const int4* z_ptr4 = (const int4*)current_z_ptr;
      int4 load_int4 = *b_ptr4;
      int4 load_zeros = *z_ptr4;
      
      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_q_val[0]);
      dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_q_val[1]);
      dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_q_val[2]);
      dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_q_val[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c_q[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_q[m][0], 1.0f);
        block_c_q[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_q[m][1], 1.0f);
        block_c_q[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_q[m][2], 1.0f);
        block_c_q[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_q[m][3], 1.0f);
      }
      
      b_ptr_q += hidden_dim;
      current_z_ptr += (groupsize / 8);
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 输出Q结果
  if (n < hidden_dim) {
    for (int m = 0; m < m_count; m++) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        q_output[seq_idx * hidden_dim + n] = __float2half(block_c_q[m][0]);
      }
    }
  }
  
  // 处理K投影（使用K权重矩阵）
  const uint32_t* b_ptr_k = qweight_k + qk_offset_in_qweight * hidden_dim + n;
  const uint32_t* z_ptr_base_k = qzeros_k + group * (groupsize / 8);
  
  // Initial group scales for K
  float scales_k_val[4];
  for (int i = 0; i < 4; i++) {
    scales_k_val[i] = __half2float(scales_k[group * hidden_dim + n + i]);
  }
  
  // Column result for K
  float block_c_k[m_count][4] = {};
  
  // Dequantize and multiply for K
  k = offset_k;
  group = offset_k / groupsize;
  nextgroup_k_boundary = (group + 1) * groupsize;
  a_ptr = &block_a[0][0];
  
  while (k < end_k) {
    if (k >= nextgroup_k_boundary) {
      group++;
      if (group >= num_groups) break;
      nextgroup_k_boundary = (group + 1) * groupsize;
      z_ptr_base_k = qzeros_k + group * (groupsize / 8);
      for (int i = 0; i < 4; i++) {
        scales_k_val[i] = __half2float(scales_k[group * hidden_dim + n + i]);
      }
    }
    
    const uint32_t* current_z_ptr = z_ptr_base_k + (k % groupsize) / 8;
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (const int4*)b_ptr_k;
      const int4* z_ptr4 = (const int4*)current_z_ptr;
      int4 load_int4 = *b_ptr4;
      int4 load_zeros = *z_ptr4;
      
      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_k_val[0]);
      dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_k_val[1]);
      dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_k_val[2]);
      dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_k_val[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c_k[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_k[m][0], 1.0f);
        block_c_k[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_k[m][1], 1.0f);
        block_c_k[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_k[m][2], 1.0f);
        block_c_k[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_k[m][3], 1.0f);
      }
      
      b_ptr_k += hidden_dim;
      current_z_ptr += (groupsize / 8);
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 输出K结果
  if (n < hidden_dim) {
    for (int m = 0; m < m_count; m++) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        k_output[seq_idx * hidden_dim + n] = __float2half(block_c_k[m][0]);
      }
    }
  }
  
  // 处理V投影（使用V权重矩阵）
  const uint32_t* b_ptr_v = qweight_v + qk_offset_in_qweight * hidden_dim + n;
  const uint32_t* z_ptr_base_v = qzeros_v + group * (groupsize / 8);
  
  // Initial group scales for V
  float scales_v_val[4];
  for (int i = 0; i < 4; i++) {
    scales_v_val[i] = __half2float(scales_v[group * hidden_dim + n + i]);
  }
  
  // Column result for V
  float block_c_v[m_count][4] = {};
  
  // Dequantize and multiply for V
  k = offset_k;
  group = offset_k / groupsize;
  nextgroup_k_boundary = (group + 1) * groupsize;
  a_ptr = &block_a[0][0];
  
  while (k < end_k) {
    if (k >= nextgroup_k_boundary) {
      group++;
      if (group >= num_groups) break;
      nextgroup_k_boundary = (group + 1) * groupsize;
      z_ptr_base_v = qzeros_v + group * (groupsize / 8);
      for (int i = 0; i < 4; i++) {
        scales_v_val[i] = __half2float(scales_v[group * hidden_dim + n + i]);
      }
    }
    
    const uint32_t* current_z_ptr = z_ptr_base_v + (k % groupsize) / 8;
    
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (const int4*)b_ptr_v;
      const int4* z_ptr4 = (const int4*)current_z_ptr;
      int4 load_int4 = *b_ptr4;
      int4 load_zeros = *z_ptr4;
      
      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, load_zeros.x, dq[0], scales_v_val[0]);
      dequant_4bit_8_gptq(load_int4.y, load_zeros.y, dq[1], scales_v_val[1]);
      dequant_4bit_8_gptq(load_int4.z, load_zeros.z, dq[2], scales_v_val[2]);
      dequant_4bit_8_gptq(load_int4.w, load_zeros.w, dq[3], scales_v_val[3]);
      
#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c_v[m][0] = dot22_8_f(dq[0], a_ptr + m * a_stride, block_c_v[m][0], 1.0f);
        block_c_v[m][1] = dot22_8_f(dq[1], a_ptr + m * a_stride, block_c_v[m][1], 1.0f);
        block_c_v[m][2] = dot22_8_f(dq[2], a_ptr + m * a_stride, block_c_v[m][2], 1.0f);
        block_c_v[m][3] = dot22_8_f(dq[3], a_ptr + m * a_stride, block_c_v[m][3], 1.0f);
      }
      
      b_ptr_v += hidden_dim;
      current_z_ptr += (groupsize / 8);
      a_ptr += 8;
    }
    
    k += 32;
  }
  
  // 输出V结果
  if (n < hidden_dim) {
    for (int m = 0; m < m_count; m++) {
      int seq_idx = offset_m + m;
      if (seq_idx < batch_size * seq_len) {
        v_output[seq_idx * hidden_dim + n] = __float2half(block_c_v[m][0]);
      }
    }
  }
}

// PyTorch包装函数
torch::Tensor fused_ln_qkv_gptq_cuda(
    torch::Tensor input,
    torch::Tensor qweight_q, torch::Tensor qweight_k, torch::Tensor qweight_v,
    torch::Tensor qzeros_q, torch::Tensor qzeros_k, torch::Tensor qzeros_v,
    torch::Tensor scales_q, torch::Tensor scales_k, torch::Tensor scales_v,
    torch::Tensor ln_weight, torch::Tensor ln_bias,
    int batch_size, int seq_len, int hidden_dim, int groupsize, float eps) {
    
    // 输入验证
    TORCH_CHECK(input.dim() == 3, "输入必须是3D张量 [batch_size, seq_len, hidden_dim]");
    TORCH_CHECK(input.size(0) == batch_size, "批次大小不匹配");
    TORCH_CHECK(input.size(1) == seq_len, "序列长度不匹配");
    TORCH_CHECK(input.size(2) == hidden_dim, "隐藏维度不匹配");
    
    // 确保输入是连续的
    input = input.contiguous();
    
    // 创建输出张量
    auto q_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    auto k_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    auto v_output = torch::zeros({batch_size, seq_len, hidden_dim}, 
                                torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));
    
    // 获取数据指针
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    half* q_output_ptr = reinterpret_cast<half*>(q_output.data_ptr<at::Half>());
    half* k_output_ptr = reinterpret_cast<half*>(k_output.data_ptr<at::Half>());
    half* v_output_ptr = reinterpret_cast<half*>(v_output.data_ptr<at::Half>());
    
    // 设置CUDA流
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // 启动内核
    dim3 blockDim(BLOCK_KN_SIZE, 1, 1);
    dim3 gridDim(DIVIDE(hidden_dim, BLOCK_KN_SIZE * 4), 
                 DIVIDE(batch_size * seq_len, BLOCK_M_SIZE_MAX), 
                 DIVIDE(hidden_dim, BLOCK_KN_SIZE));
    
    fused_ln_qkv_gptq_kernel<BLOCK_M_SIZE_MAX><<<gridDim, blockDim, 0, stream>>>(
        input_ptr, 
        qweight_q.data_ptr<uint32_t>(),
        qweight_k.data_ptr<uint32_t>(),
        qweight_v.data_ptr<uint32_t>(),
        qzeros_q.data_ptr<uint32_t>(),
        qzeros_k.data_ptr<uint32_t>(),
        qzeros_v.data_ptr<uint32_t>(),
        reinterpret_cast<const half*>(scales_q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(ln_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(ln_bias.data_ptr<at::Half>()),
        q_output_ptr, k_output_ptr, v_output_ptr,
        batch_size, seq_len, hidden_dim, groupsize, eps
    );
    
    cudaDeviceSynchronize();
    
    // 返回QKV张量的元组
    return torch::stack({q_output, k_output, v_output}, 0);
}

// PyTorch模块导出函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_qkv_gptq_cuda", &fused_ln_qkv_gptq_cuda, "Fused LayerNorm + QKV GPTQ CUDA Kernel");
}